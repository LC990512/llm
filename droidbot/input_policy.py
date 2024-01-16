import sys
import json
import re
import logging
import random
import openai
import time
import itertools

from abc import abstractmethod

from .input_event import *
from .utg import UTG

# Max number of restarts
MAX_NUM_RESTARTS = 5
# Max number of steps outside the app
MAX_NUM_STEPS_OUTSIDE = 5
MAX_NUM_STEPS_OUTSIDE_KILL = 10
# Max number of replay tries
MAX_REPLY_TRIES = 5

# Some input event flags
EVENT_FLAG_STARTED = "+started"
EVENT_FLAG_START_APP = "+start_app"
EVENT_FLAG_STOP_APP = "+stop_app"
EVENT_FLAG_EXPLORE = "+explore"
EVENT_FLAG_NAVIGATE = "+navigate"
EVENT_FLAG_TOUCH = "+touch"

# Policy taxanomy
POLICY_NAIVE_DFS = "dfs_naive"
POLICY_GREEDY_DFS = "dfs_greedy"
POLICY_NAIVE_BFS = "bfs_naive"
POLICY_GREEDY_BFS = "bfs_greedy"
POLICY_REPLAY = "replay"
POLICY_MANUAL = "manual"
POLICY_MONKEY = "monkey"
POLICY_TASK = "task"
POLICY_NONE = "none"
POLICY_MEMORY_GUIDED = "memory_guided"  # implemented in input_policy2


class InputInterruptedException(Exception):
    pass


class InputPolicy(object):
    """
    This class is responsible for generating events to stimulate more app behaviour
    It should call AppEventManager.send_event method continuously
    """

    def __init__(self, device, app):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.app = app
        self.action_count = 0
        self.master = None
       

    def start(self, input_manager):
        """
        start producing events
        :param input_manager: instance of InputManager
        """
        self.action_count = 0
        while input_manager.enabled and self.action_count < input_manager.event_count:
            try:
                # # make sure the first event is go to HOME screen
                # # the second event is to start the app
                # if self.action_count == 0 and self.master is None:
                #     event = KeyEvent(name="HOME")
                # elif self.action_count == 1 and self.master is None:
                #     event = IntentEvent(self.app.get_start_intent())
                if self.action_count < 0 and self.master is None:
                    event = KillAppEvent(app=self.app)
                else:
                    finish, event = self.generate_event()
                input_manager.add_event(event)
            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.warning("stop sending events: %s" % e)
                break
            # except RuntimeError as e:
            #     self.logger.warning(e.message)
            #     break
            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback
                traceback.print_exc()
                continue
            self.action_count += 1

    @abstractmethod
    def generate_event(self):
        """
        generate an event
        @return:
        """
        pass


class NoneInputPolicy(InputPolicy):
    """
    do not send any event
    """

    def __init__(self, device, app):
        super(NoneInputPolicy, self).__init__(device, app)

    def generate_event(self):
        """
        generate an event
        @return:
        """
        return None


class UtgBasedInputPolicy(InputPolicy):
    """
    state-based input policy
    """

    def __init__(self, device, app, random_input):
        super(UtgBasedInputPolicy, self).__init__(device, app)
        self.random_input = random_input
        self.script = None
        self.master = None
        self.script_events = []
        self.last_event = None
        self.last_state = None
        self.current_state = None
        self.utg = UTG(device=device, app=app, random_input=random_input)
        self.script_event_idx = 0
        if self.device.humanoid is not None:
            self.humanoid_view_trees = []
            self.humanoid_events = []

    def generate_event(self):
        """
        generate an event
        @return:
        """

        # Get current device state
        self.current_state = self.device.get_current_state()
        if self.current_state is None:
            import time
            time.sleep(5)
            return KeyEvent(name="BACK")

        self.__update_utg()

        # update last view trees for humanoid
        if self.device.humanoid is not None:
            self.humanoid_view_trees = self.humanoid_view_trees + [self.current_state.view_tree]
            if len(self.humanoid_view_trees) > 4:
                self.humanoid_view_trees = self.humanoid_view_trees[1:]

        event = None

        # if the previous operation is not finished, continue
        if len(self.script_events) > self.script_event_idx:
            event = self.script_events[self.script_event_idx].get_transformed_event(self)
            finish = 0
            self.script_event_idx += 1

        # First try matching a state defined in the script
        if event is None and self.script is not None:
            operation = self.script.get_operation_based_on_state(self.current_state)
            if operation is not None:
                self.script_events = operation.events
                # restart script
                event = self.script_events[0].get_transformed_event(self)
                finish = 0
                self.script_event_idx = 1

        if event is None:
            finish, event = self.generate_event_based_on_utg()
            print(f"lccc generate_event finish:[ {finish} ], event: [ {event} ]")

        if finish == -1:
            print(f"lccc generate_event finish == -1")
            return finish, event

        # update last events for humanoid
        if self.device.humanoid is not None:
            self.humanoid_events = self.humanoid_events + [event]
            if len(self.humanoid_events) > 3:
                self.humanoid_events = self.humanoid_events[1:]

        self.last_state = self.current_state
        self.last_event = event
        return finish, event

    def __update_utg(self):
        self.utg.add_transition(self.last_event, self.last_state, self.current_state)

    @abstractmethod
    def generate_event_based_on_utg(self):
        """
        generate an event based on UTG
        :return: InputEvent
        """
        pass


class UtgNaiveSearchPolicy(UtgBasedInputPolicy):
    """
    depth-first strategy to explore UFG (old)
    """

    def __init__(self, device, app, random_input, search_method):
        super(UtgNaiveSearchPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.explored_views = set()
        self.state_transitions = set()
        self.search_method = search_method

        self.last_event_flag = ""
        self.last_event_str = None
        self.last_state = None

        self.preferred_buttons = ["yes", "ok", "activate", "detail", "more", "access",
                                  "allow", "check", "agree", "try", "go", "next"]

    def generate_event_based_on_utg(self):
        """
        generate an event based on current device state
        note: ensure these fields are properly maintained in each transaction:
          last_event_flag, last_touched_view, last_state, exploited_views, state_transitions
        @return: InputEvent
        """
        self.save_state_transition(self.last_event_str, self.last_state, self.current_state)

        if self.device.is_foreground(self.app):
            # the app is in foreground, clear last_event_flag
            self.last_event_flag = EVENT_FLAG_STARTED
        else:
            number_of_starts = self.last_event_flag.count(EVENT_FLAG_START_APP)
            # If we have tried too many times but the app is still not started, stop DroidBot
            if number_of_starts > MAX_NUM_RESTARTS:
                raise InputInterruptedException("The app cannot be started.")

            # if app is not started, try start it
            if self.last_event_flag.endswith(EVENT_FLAG_START_APP):
                # It seems the app stuck at some state, and cannot be started
                # just pass to let viewclient deal with this case
                self.logger.info("The app had been restarted %d times.", number_of_starts)
                self.logger.info("Trying to restart app...")
                pass
            else:
                start_app_intent = self.app.get_start_intent()

                self.last_event_flag += EVENT_FLAG_START_APP
                self.last_event_str = EVENT_FLAG_START_APP
                return IntentEvent(start_app_intent)

        # select a view to click
        view_to_touch = self.select_a_view(self.current_state)

        # if no view can be selected, restart the app
        if view_to_touch is None:
            stop_app_intent = self.app.get_stop_intent()
            self.last_event_flag += EVENT_FLAG_STOP_APP
            self.last_event_str = EVENT_FLAG_STOP_APP
            return IntentEvent(stop_app_intent)

        view_to_touch_str = view_to_touch['view_str']
        if view_to_touch_str.startswith('BACK'):
            result = KeyEvent('BACK')
        else:
            result = TouchEvent(view=view_to_touch)

        self.last_event_flag += EVENT_FLAG_TOUCH
        self.last_event_str = view_to_touch_str
        self.save_explored_view(self.current_state, self.last_event_str)
        return result

    def select_a_view(self, state):
        """
        select a view in the view list of given state, let droidbot touch it
        @param state: DeviceState
        @return:
        """
        views = []
        for view in state.views:
            if view['enabled'] and len(view['children']) == 0:
                views.append(view)

        if self.random_input:
            random.shuffle(views)

        # add a "BACK" view, consider go back first/last according to search policy
        mock_view_back = {'view_str': 'BACK_%s' % state.foreground_activity,
                          'text': 'BACK_%s' % state.foreground_activity}
        if self.search_method == POLICY_NAIVE_DFS:
            views.append(mock_view_back)
        elif self.search_method == POLICY_NAIVE_BFS:
            views.insert(0, mock_view_back)

        # first try to find a preferable view
        for view in views:
            view_text = view['text'] if view['text'] is not None else ''
            view_text = view_text.lower().strip()
            if view_text in self.preferred_buttons \
                    and (state.foreground_activity, view['view_str']) not in self.explored_views:
                self.logger.info("selected an preferred view: %s" % view['view_str'])
                return view

        # try to find a un-clicked view
        for view in views:
            if (state.foreground_activity, view['view_str']) not in self.explored_views:
                self.logger.info("selected an un-clicked view: %s" % view['view_str'])
                return view

        # if all enabled views have been clicked, try jump to another activity by clicking one of state transitions
        if self.random_input:
            random.shuffle(views)
        transition_views = {transition[0] for transition in self.state_transitions}
        for view in views:
            if view['view_str'] in transition_views:
                self.logger.info("selected a transition view: %s" % view['view_str'])
                return view

        # no window transition found, just return a random view
        # view = views[0]
        # self.logger.info("selected a random view: %s" % view['view_str'])
        # return view

        # DroidBot stuck on current state, return None
        self.logger.info("no view could be selected in state: %s" % state.tag)
        return None

    def save_state_transition(self, event_str, old_state, new_state):
        """
        save the state transition
        @param event_str: str, representing the event cause the transition
        @param old_state: DeviceState
        @param new_state: DeviceState
        @return:
        """
        if event_str is None or old_state is None or new_state is None:
            return
        if new_state.is_different_from(old_state):
            self.state_transitions.add((event_str, old_state.tag, new_state.tag))

    def save_explored_view(self, state, view_str):
        """
        save the explored view
        @param state: DeviceState, where the view located
        @param view_str: str, representing a view
        @return:
        """
        if not state:
            return
        state_activity = state.foreground_activity
        self.explored_views.add((state_activity, view_str))


class UtgGreedySearchPolicy(UtgBasedInputPolicy):
    """
    DFS/BFS (according to search_method) strategy to explore UFG (new)
    """

    def __init__(self, device, app, random_input, search_method):
        super(UtgGreedySearchPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search_method = search_method

        self.preferred_buttons = ["yes", "ok", "activate", "detail", "more", "access",
                                  "allow", "check", "agree", "try", "go", "next"]

        self.__nav_target = None
        self.__nav_num_steps = -1
        self.__num_restarts = 0
        self.__num_steps_outside = 0
        self.__event_trace = ""
        self.__missed_states = set()
        self.__random_explore = False

    def generate_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        current_state = self.current_state
        self.logger.info("Current state: %s" % current_state.state_str)
        if current_state.state_str in self.__missed_states:
            self.__missed_states.remove(current_state.state_str)

        if current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            start_app_intent = self.app.get_start_intent()

            # It seems the app stucks at some state, has been
            # 1) force stopped (START, STOP)
            #    just start the app again by increasing self.__num_restarts
            # 2) started at least once and cannot be started (START)
            #    pass to let viewclient deal with this case
            # 3) nothing
            #    a normal start. clear self.__num_restarts.

            if self.__event_trace.endswith(EVENT_FLAG_START_APP + EVENT_FLAG_STOP_APP) \
                    or self.__event_trace.endswith(EVENT_FLAG_START_APP):
                self.__num_restarts += 1
                self.logger.info("The app had been restarted %d times.", self.__num_restarts)
            else:
                self.__num_restarts = 0

            # pass (START) through
            if not self.__event_trace.endswith(EVENT_FLAG_START_APP):
                if self.__num_restarts > MAX_NUM_RESTARTS:
                    # If the app had been restarted too many times, enter random mode
                    msg = "The app had been restarted too many times. Entering random mode."
                    self.logger.info(msg)
                    self.__random_explore = True
                else:
                    # Start the app
                    self.__event_trace += EVENT_FLAG_START_APP
                    self.logger.info("Trying to start the app...")
                    return IntentEvent(intent=start_app_intent)

        elif current_state.get_app_activity_depth(self.app) > 0:
            # If the app is in activity stack but is not in foreground
            self.__num_steps_outside += 1

            if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                # If the app has not been in foreground for too long, try to go back
                if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE_KILL:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    go_back_event = KeyEvent(name="BACK")
                self.__event_trace += EVENT_FLAG_NAVIGATE
                self.logger.info("Going back to the app...")
                return go_back_event
        else:
            # If the app is in foreground
            self.__num_steps_outside = 0

        # Get all possible input events
        possible_events = current_state.get_possible_input()

        if self.random_input:
            random.shuffle(possible_events)

        if self.search_method == POLICY_GREEDY_DFS:
            possible_events.append(KeyEvent(name="BACK"))
        elif self.search_method == POLICY_GREEDY_BFS:
            possible_events.insert(0, KeyEvent(name="BACK"))

        # get humanoid result, use the result to sort possible events
        # including back events
        if self.device.humanoid is not None:
            possible_events = self.__sort_inputs_by_humanoid(possible_events)

        # If there is an unexplored event, try the event first
        for input_event in possible_events:
            if not self.utg.is_event_explored(event=input_event, state=current_state):
                self.logger.info("Trying an unexplored event.")
                self.__event_trace += EVENT_FLAG_EXPLORE
                return input_event

        target_state = self.__get_nav_target(current_state)
        if target_state:
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=target_state)
            if navigation_steps and len(navigation_steps) > 0:
                self.logger.info("Navigating to %s, %d steps left." % (target_state.state_str, len(navigation_steps)))
                self.__event_trace += EVENT_FLAG_NAVIGATE
                return navigation_steps[0][1]

        if self.__random_explore:
            self.logger.info("Trying random event.")
            random.shuffle(possible_events)
            return possible_events[0]

        # If couldn't find a exploration target, stop the app
        stop_app_intent = self.app.get_stop_intent()
        self.logger.info("Cannot find an exploration target. Trying to restart app...")
        self.__event_trace += EVENT_FLAG_STOP_APP
        return IntentEvent(intent=stop_app_intent)

    def __sort_inputs_by_humanoid(self, possible_events):
        if sys.version.startswith("3"):
            from xmlrpc.client import ServerProxy
        else:
            from xmlrpclib import ServerProxy
        proxy = ServerProxy("http://%s/" % self.device.humanoid)
        request_json = {
            "history_view_trees": self.humanoid_view_trees,
            "history_events": [x.__dict__ for x in self.humanoid_events],
            "possible_events": [x.__dict__ for x in possible_events],
            "screen_res": [self.device.display_info["width"],
                           self.device.display_info["height"]]
        }
        result = json.loads(proxy.predict(json.dumps(request_json)))
        new_idx = result["indices"]
        text = result["text"]
        new_events = []

        # get rid of infinite recursive by randomizing first event
        if not self.utg.is_state_reached(self.current_state):
            new_first = random.randint(0, len(new_idx) - 1)
            new_idx[0], new_idx[new_first] = new_idx[new_first], new_idx[0]

        for idx in new_idx:
            if isinstance(possible_events[idx], SetTextEvent):
                possible_events[idx].text = text
            new_events.append(possible_events[idx])
        return new_events

    def __get_nav_target(self, current_state):
        # If last event is a navigation event
        if self.__nav_target and self.__event_trace.endswith(EVENT_FLAG_NAVIGATE):
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=self.__nav_target)
            if navigation_steps and 0 < len(navigation_steps) <= self.__nav_num_steps:
                # If last navigation was successful, use current nav target
                self.__nav_num_steps = len(navigation_steps)
                return self.__nav_target
            else:
                # If last navigation was failed, add nav target to missing states
                self.__missed_states.add(self.__nav_target.state_str)

        reachable_states = self.utg.get_reachable_states(current_state)
        if self.random_input:
            random.shuffle(reachable_states)

        for state in reachable_states:
            # Only consider foreground states
            if state.get_app_activity_depth(self.app) != 0:
                continue
            # Do not consider missed states
            if state.state_str in self.__missed_states:
                continue
            # Do not consider explored states
            if self.utg.is_state_explored(state):
                continue
            self.__nav_target = state
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=self.__nav_target)
            if len(navigation_steps) > 0:
                self.__nav_num_steps = len(navigation_steps)
                return state

        self.__nav_target = None
        self.__nav_num_steps = -1
        return None


class UtgReplayPolicy(InputPolicy):
    """
    Replay DroidBot output generated by UTG policy
    """

    def __init__(self, device, app, replay_output):
        super(UtgReplayPolicy, self).__init__(device, app)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.replay_output = replay_output

        import os
        event_dir = os.path.join(replay_output, "events")
        self.event_paths = sorted([os.path.join(event_dir, x) for x in
                                   next(os.walk(event_dir))[2]
                                   if x.endswith(".json")])
        # skip HOME and start app intent
        self.device = device
        self.app = app
        self.event_idx = 2
        self.num_replay_tries = 0
        self.utg = UTG(device=device, app=app, random_input=None)
        self.last_event = None
        self.last_state = None
        self.current_state = None

    def generate_event(self):
        """
        generate an event based on replay_output
        @return: InputEvent
        """
        import time
        while self.event_idx < len(self.event_paths) and \
              self.num_replay_tries < MAX_REPLY_TRIES:
            self.num_replay_tries += 1
            current_state = self.device.get_current_state()
            if current_state is None:
                time.sleep(5)
                self.num_replay_tries = 0
                return KeyEvent(name="BACK")

            curr_event_idx = self.event_idx
            self.__update_utg()
            while curr_event_idx < len(self.event_paths):
                event_path = self.event_paths[curr_event_idx]
                with open(event_path, "r") as f:
                    curr_event_idx += 1

                    try:
                        event_dict = json.load(f)
                    except Exception as e:
                        self.logger.info("Loading %s failed" % event_path)
                        continue

                    if event_dict["start_state"] != current_state.state_str:
                        continue
                    if not self.device.is_foreground(self.app):
                        # if current app is in background, bring it to foreground
                        component = self.app.get_package_name()
                        if self.app.get_main_activity():
                            component += "/%s" % self.app.get_main_activity()
                        return IntentEvent(Intent(suffix=component))
                    
                    self.logger.info("Replaying %s" % event_path)
                    self.event_idx = curr_event_idx
                    self.num_replay_tries = 0
                    # return InputEvent.from_dict(event_dict["event"])
                    event = InputEvent.from_dict(event_dict["event"])
                    self.last_state = self.current_state
                    self.last_event = event
                    return event                    

            time.sleep(5)

        # raise InputInterruptedException("No more record can be replayed.")
    def __update_utg(self):
        self.utg.add_transition(self.last_event, self.last_state, self.current_state)


class ManualPolicy(UtgBasedInputPolicy):
    """
    manually explore UFG
    """

    def __init__(self, device, app):
        super(ManualPolicy, self).__init__(device, app, False)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.__first_event = True

    def generate_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        if self.__first_event:
            self.__first_event = False
            self.logger.info("Trying to start the app...")
            start_app_intent = self.app.get_start_intent()
            return IntentEvent(intent=start_app_intent)
        else:
            return ManualEvent()


class TaskPolicy(UtgBasedInputPolicy):

    def __init__(self, device, app, random_input, step, extracted_info, api):
        super(TaskPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.__nav_target = None
        self.__nav_num_steps = -1
        self.step = step
        #self.task = extracted_info[step-1]['task']
        self.task = "start"
        self.extracted_info = extracted_info
        self.__num_restarts = 0
        self.__num_steps_outside = 0
        self.__event_trace = ""
        self.__missed_states = set()
        self.__random_explore = random_input
        self.__action_history = []
        self.api = api


    #lccc
    def start(self, input_manager):
        """
        start producing events
        :param input_manager: instance of InputManager
        """
        self.action_count = 0
        #self.step = len(self.extracted_info)
        max_step = len(self.extracted_info) * 3
        while input_manager.enabled and self.action_count < max_step:#input_manager.event_count:
            try:
                if self.action_count == 0 and self.master is None:
                    event = KillAppEvent(app=self.app)
                    finish = 0
                else:
                    #lccc 先忽略system enter 和 layout
                    if ("system enter" in self.task.lower()) or ("layout" in self.task.lower()):
                        finish = -1
                    else:
                        finish, event = self.generate_event()

                print(f'lccc start: finish [{finish}] / event[{event}] / task[{self.task}]')

                if finish != -1:
                    input_manager.add_event(event)
                    time.sleep(5)
                if finish != 0:
                    if  self.step < len(self.extracted_info):
                        self.step += 1
                        self.task = self.extracted_info[self.step-1]['task']
                    else:
                        break
            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.warning("stop sending events: %s" % e)
                break
            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback
                traceback.print_exc()
                continue
            self.action_count += 1
        

    def generate_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        current_state = self.current_state
        self.logger.info("Current state: %s" % current_state.state_str)
        if current_state.state_str in self.__missed_states:
            self.__missed_states.remove(current_state.state_str)

        if current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            start_app_intent = self.app.get_start_intent()

            # It seems the app stucks at some state, has been
            # 1) force stopped (START, STOP)
            #    just start the app again by increasing self.__num_restarts
            # 2) started at least once and cannot be started (START)
            #    pass to let viewclient deal with this case
            # 3) nothing
            #    a normal start. clear self.__num_restarts.

            if self.__event_trace.endswith(EVENT_FLAG_START_APP + EVENT_FLAG_STOP_APP) \
                    or self.__event_trace.endswith(EVENT_FLAG_START_APP):
                self.__num_restarts += 1
                self.logger.info("The app had been restarted %d times.", self.__num_restarts)
            else:
                self.__num_restarts = 0

            # pass (START) through
            if not self.__event_trace.endswith(EVENT_FLAG_START_APP):
                if self.__num_restarts > MAX_NUM_RESTARTS:
                    # If the app had been restarted too many times, enter random mode
                    msg = "The app had been restarted too many times. Entering random mode."
                    self.logger.info(msg)
                    self.__random_explore = True
                else:
                    # Start the app
                    print("lc---------------------------------------------first")
                    self.__event_trace += EVENT_FLAG_START_APP
                    self.logger.info("Trying to start the app...")
                    self.__action_history = [f'- start the app {self.app.app_name}']
                    return 1, IntentEvent(intent=start_app_intent)

        elif current_state.get_app_activity_depth(self.app) > 0:
            # If the app is in activity stack but is not in foreground
            self.__num_steps_outside += 1

            if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                # If the app has not been in foreground for too long, try to go back
                if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE_KILL:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    go_back_event = KeyEvent(name="BACK")
                self.__event_trace += EVENT_FLAG_NAVIGATE
                self.logger.info("Going back to the app...")
                self.__action_history.append('- go back')
                return 1, go_back_event
        else:
            # If the app is in foreground
            self.__num_steps_outside = 0
        
        #lccc
        if self.extracted_info[self.step-1]['status'] == -1:
            finish, action, candidate_actions = self._get_action_with_LLM(current_state, self.__action_history)
        else:
            finish, action, candidate_actions = self._get_action_with_match(current_state, self.__action_history)
            if action is None:
                self.extracted_info[self.step-1]['status'] = -1
                finish, action, candidate_actions = self._get_action_with_LLM(current_state, self.__action_history)

        
        if action is not None:
            desc = current_state.get_action_desc(action)
            self.__action_history.append(desc)
            print(f"lccc action: [ {action} ] desc: [ {desc} ] task: [ {self.task}]")
            return finish, action

        if (finish != -1) and (self.__random_explore):
            self.logger.info("Trying random event.")
            action = random.choice(candidate_actions)
            self.__action_history.append(current_state.get_action_desc(action))
            return finish, action

        if (finish == -1):
            return finish, action
        
        # If couldn't find a exploration target, stop the app
        stop_app_intent = self.app.get_stop_intent()
        self.logger.info("Cannot find an exploration target. Trying to restart app...")
        self.__action_history.append('- stop the app')
        self.__event_trace += EVENT_FLAG_STOP_APP
        return finish, IntentEvent(intent=stop_app_intent)

    #lccc
    def make_openai_request(self, messages):
        api_keys = self.api
        while True:
            current_time = time.time()
            for api_key_info in api_keys:
                # 检查自上次使用该密钥以来是否已经过去了30秒
                print("lccc:", api_key_info, current_time - api_key_info['last_used'])
                if current_time - api_key_info['last_used'] > 30:
                    try:
                        openai.api_key = api_key_info['key']
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo-0613",
                            max_tokens=1024,
                            temperature=0.8,
                            messages = messages
                        )
                        api_key_info['last_used'] = time.time()  # 更新最后使用时间
                        return response
                    except openai.error.RateLimitError:
                        # 如果遇到速率限制错误，记录时间并尝试下一个密钥
                        api_key_info['last_used'] = time.time()
                        continue
            # 如果所有密钥都未能使用，暂停一段时间再重试
            time.sleep(10)

    def _query_llm(self, prompt):
        messages=[{"role": "user", "content": prompt}]
        response = self.make_openai_request(messages)
        real_ans = response['choices'][0]['message']['content']
        return real_ans
          
    def if_action(self, now_action):
        print(f'lccc if_action candidate: {now_action}, {type(now_action)}')

        #system enter怎么判断？除了click/edit以外的怎么判断？
        if "system" in self.task:
            return 0
        #例 Edit a view "log in password" with "research"
        flag = 1
        text = ""

        #action = Edit
        action = self.task.split()[0].lower()
        print(f'lccc if_action action: {action}')
        if hasattr(now_action, 'view'):
            view = now_action.view
            clickable = view['clickable']
            editable = view['editable']
            print(f'lccc if_action action:  clickable: {clickable} editable:{editable}')
        #else可能不对,有一些没有view的event我不知道是什么
        else:
            return 0, text
        
        if action == "click":
            if not clickable:
                return 0, text
        elif action == "edit":
            if not editable:
                return 0, text

        if "with" in self.task:
            text = self.task.split("with")[1].split("\"")[1]

        #id = log in password
        id = self.task.replace(",", "").split("\"")[1].lower().split()
        item = ""
        if view['resource_id']:
            item += view['resource_id'] + ','
        if view['class']:
            item += view['class'] + ','
        if view['text']:
            item += view['text'] + ','
        if view['content_description']:
            item += view['content_description']
        if item == "":
            return 0, text
        
        for _id in id:
            print(f'lccc if_action _id: {_id} item: {item}')
            if _id == text:
                continue
            if _id not in item.lower():
                return 0, text

        #text = "" / research
        
        return flag, text
        
    def _get_action_with_match(self, current_state, action_history):
        finish = 0
        selected_action = None
        view_descs, candidate_actions = current_state.get_described_actions()
        state_desc = '\n'.join(view_descs)

        if "system" in self.task:
            return finish, selected_action, candidate_actions
        
        print(f'lccc _get_action_with_match candidate: {len(view_descs)}——{len(candidate_actions)}——{self.task}')
        for idx in range(0, len(candidate_actions)):
            desc = state_desc.split("("+str(idx)+")")[0]
            if idx > 0:
                desc = desc.split("("+str(idx-1)+")")[1]
            desc = desc.replace('\n', '')
            desc += "("+str(idx)+")"

            print(f'lccc _get_action_with_match candidate: {idx}——{desc}——{self.task}')
            #if_action判断当前候选action是否匹配
            flag, text = self.if_action(candidate_actions[idx])
            print(f'lccc _get_action_with_match if_else: {idx}——{flag}')
            input()
            if flag == 1:
                selected_action = candidate_actions[idx]
                if text != "":
                    selected_action.text = text
                finish = 1
                print(f'lccc _get_action_with_match end: {idx}——{desc}')
                break
        return finish, selected_action, candidate_actions

    def _get_action_with_LLM(self, current_state, action_history):
        app = self.extracted_info[self.step-1]['app'].split("/")[1].split(".")[0]
        func = self.extracted_info[self.step-1]['function']
        view_descs, candidate_actions = current_state.get_described_actions()

        # First, determine whether the task has already been completed.
        task_prompt = f"I am currently focused on a specific step in my test case, identified as '{self.task}'. This step might correspond to one or several of the actions I have already executed."
        history_prompt = f'Executed Actions: \n ' + ';\n '.join(action_history)
        question = f"Question: Based on the actions I have executed so far, have I completed the step '{self.task}'? Please only return 'yes' or 'no'. A 'yes' means the step has been completed, and a 'no' means I need to continue exploring further actions."
        prompt = f'{task_prompt}\n{history_prompt}\n{question}'
        print(prompt)
        response = self._query_llm(prompt)
        print(f'response: {response}')

        response = input()

        if ("yes" in response.lower()):
            finish = -1
            print(f"Seems the task is completed. Press Enter to continue...")
            return finish, None, candidate_actions
    
        # Second, if not finished, then provide the next action.
        #task_prompt = f"I'm using a smartphone to '{self.task}' in the '{app}' app. My current task requires completing the step '{self.task}'"
        #task_prompt = f"I am working on a test case that involves multiple steps to complete the testing of the '{func}' feature, and I am currently at the step '{self.task}'. "
        task_prompt = f"I am working on a test case for the '{func}' feature in the '{app}' app. At this stage, I need to choose the next step that will effectively advance the testing process. I've already completed some actions."
        task_prompt += f"I currently need to choose an action to help me complete this step of {self.task}."
        #history_prompt = f'I have already completed the following steps, which should not be performed again: \n ' + ';\n '.join(action_history)
        #history_prompt = f'Below is a list of actions I have already executed, which should not be performed again: \n' + ';\n '.join(action_history)
        history_prompt = 'Completed Actions (please do not suggest these again): \n' + ';\n '.join(action_history)
        state_prompt = 'Current State with Available UI Views and Actions (with Action ID):\n ' + ';\n '.join(view_descs)
        #state_prompt = 'The current state has the following UI views and corresponding actions, with action id in parentheses:\n ' + ';\n '.join(view_descs)
        #question = "Which action should I choose next? Please only return the action id and nothing else.\n"
        #question = "Based on the information provided, I need a suggestion: Which action (please only return the action's ID) should I execute at this step on the current page to effectively complete my task?"
        question = f"Given these options, which action (identified by the Action ID) should I perform next to effectively continue testing the '{func}' feature? Please do not suggest any actions that I have already completed, only return the action id."
        #prompt = f'{task_prompt}\n{state_prompt}\n{history_prompt}\n{question}'
        prompt = f'{task_prompt}\n{history_prompt}\n{state_prompt}\n{question}'
        
        # extra_prompt = f"I'm using a smartphone to test the '{func}' functionality in the '{app}' app. I need to determine the appropriate UI action to perform next based on the current state of the app. My objective is to ensure that the chosen action is logically aligned with the potential functionality. \n"
        # prompt = f'{extra_prompt}\n{history_prompt}\n{state_prompt}\n{question}'
        print(prompt)
        response = self._query_llm(prompt)
        print(f'response: {response}')

        response = input()

    
        match = re.search(r'\d+', response)
        finish = 0
        if not match:
            selected_action = candidate_actions[-1]
            return finish, selected_action, candidate_actions
        
        idx = int(match.group(0))
        print(f"lccc idx: {idx}")

        selected_action = candidate_actions[idx]
        if isinstance(selected_action, SetTextEvent):
            view_text = current_state.get_view_desc(selected_action.view)
            question = f'I have chosen the action of {view_text}. So I need to type something into the dialog box. What text should I enter to the {view_text}? Just return the text need enter and nothing else.'
            #prompt = f'{task_prompt}\n{state_prompt}\n{question}'
            prompt = f'{task_prompt}\n{history_prompt}\n{question}'
            print(prompt)
            response = self._query_llm(prompt)
            selected_action.text = response
            print(f'response: {response}')
            if "\"" in response:
                selected_action.text = re.findall(r'"([^"]*)"', response)[-1]
            print(f'response: {selected_action.text}')
            selected_action.text = input()
        print(f"lccc _get_action_with_LLM finish: {finish}; selected_action: {selected_action}")
        return finish, selected_action, candidate_actions
        # except:
        #     import traceback
        #     traceback.print_exc()
        #     return None, candidate_actions

