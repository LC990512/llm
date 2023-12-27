# helper file of droidbot
# it parses command arguments and send the options to droidbot
import argparse
import re
import os
from droidbot import input_manager
from droidbot import input_policy
from droidbot import env_manager
from droidbot import DroidBot
from droidbot.droidmaster import DroidMaster


def parse_args(extracted_info):
    """
    parse command line input
    generate options including host name, port number
    """
    parser = argparse.ArgumentParser(description="Start DroidBot to test an Android app.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", action="store", dest="device_serial", required=False,
                        help="The serial number of target device (use `adb devices` to find)")
    
    #lccc-1
    parser.add_argument("-a", action="store", dest="apk_path",
                        help="The file path to target APK", default=extracted_info[0]['app'])
    parser.add_argument("-o", action="store", dest="output_dir", required=True,
                        help="directory of output", default="lc")
    parser.add_argument("-task", action="store", dest="task",
                        help="the task to execute, in natural language", default=extracted_info[0]['task'])
    parser.add_argument("-step", action="store", dest="step",
                        help="whether generation is required for this task", default=1)
    parser.add_argument("-extracted_info", action="store", dest="extracted_info",
                        help="The extracted information of test case steps", default=extracted_info)

    #lccc-2

    parser.add_argument("-script", action="store", dest="script_path",
                        help="Use a script to customize input for certain states.")
    parser.add_argument("-count", action="store", dest="count", default=input_manager.DEFAULT_EVENT_COUNT, type=int,
                        help="Number of events to generate in total. Default: %d" % input_manager.DEFAULT_EVENT_COUNT)
    parser.add_argument("-interval", action="store", dest="interval", default=input_manager.DEFAULT_EVENT_INTERVAL,
                        type=int,
                        help="Interval in seconds between each two events. Default: %d" % input_manager.DEFAULT_EVENT_INTERVAL)
    parser.add_argument("-timeout", action="store", dest="timeout", default=input_manager.DEFAULT_TIMEOUT, type=int,
                        help="Timeout in seconds, -1 means unlimited. Default: %d" % input_manager.DEFAULT_TIMEOUT)
    parser.add_argument("-debug", action="store_true", dest="debug_mode",
                        help="Run in debug mode (dump debug messages).")
    parser.add_argument("-keep_app", action="store_true", dest="keep_app",
                        help="Keep the app on the device after testing.")
    parser.add_argument("-keep_env", action="store_true", dest="keep_env",
                        help="Keep the test environment (eg. minicap and accessibility service) after testing.")
    parser.add_argument("-grant_perm", action="store_true", dest="grant_perm",
                        help="Grant all permissions while installing. Useful for Android 6.0+.")
    parser.add_argument("-is_emulator", action="store_true", dest="is_emulator",
                        help="Declare the target device to be an emulator, which would be treated specially by DroidBot.")
    parser.add_argument("-accessibility_auto", action="store_true", dest="enable_accessibility_hard",
                        help="Enable the accessibility service automatically even though it might require device restart\n(can be useful for Android API level < 23).")
    parser.add_argument("-ignore_ad", action="store_true", dest="ignore_ad",
                        help="Ignore Ad views by checking resource_id.")
    options = parser.parse_args()
    # print options
    return options

def process_line(line):
    if line.endswith('.'):
        return line[:-1]
    return line

def process_files(directory, filename):
    # 正则表达式，用于匹配特定格式的行
    file_prefix = filename.split("_")[0]
    file_suffix = filename.split("_")[1].split(".")[0]
    with open("../step1/result_paser/{}".format(filename), "r") as file:
        general_step = file.read()
    pattern = re.compile(r'Test Step (\d+): \((Event|Assertion)\) (.+)')
    pattern = re.compile(r'Test Step (\d+):\.\s*\((Event|Assertion)\)\s+(.+)')

    extracted_info = []
    with open(os.path.join(directory, filename), 'r') as file:
        # extracted_info.append({
        #     'app': "apps/" + file_prefix + ".apk",
        #     'function': file_suffix,
        #     'step_number': 1,
        #     'event_or_assertion': "Event",
        #     'task': "Open the '{}' app".format(file_prefix),
        #     'status': -1
        # })
        for line in file:
            match = pattern.match(line)
            if match:
                # 提取数字，Event/Assertion 和 task
                step_number = int(match.group(1))
                event_or_assertion = match.group(2)
                task = match.group(3)
                status = 1 
            else:
                step_number = int(line.split(":")[0].split("Test Step ")[1])
                event_or_assertion = general_step.splitlines()[step_number-1].split(")")[0].split("(")[1]
                task = general_step.splitlines()[step_number-1].split(") ")[1]
                status = -1

            # 将提取的信息添加到列表
            extracted_info.append({
                'app': "apps/" + file_prefix + ".apk",
                'function': file_suffix,
                'step_number': step_number,
                'event_or_assertion': event_or_assertion,
                'task': process_line(task) + " in {} app.".format(file_prefix),
                'status': status
            })
    return extracted_info

def explore(extracted_info):
    opts = parse_args(extracted_info)

    if not os.path.exists(opts.apk_path):
        print("APK does not exist.")
        return
    print(opts)
    print("-----------------------------")
    droidbot = DroidBot(
        app_path=opts.apk_path,
        device_serial=opts.device_serial,
        task=opts.task,
        step=opts.step,
        extracted_info=opts.extracted_info,
        is_emulator=opts.is_emulator,
        output_dir=opts.output_dir,
        env_policy=env_manager.POLICY_NONE,
        policy_name=input_manager.POLICY_TASK,
        script_path=opts.script_path,
        event_interval=opts.interval,
        timeout=opts.timeout,
        event_count=opts.count,
        debug_mode=opts.debug_mode,
        keep_app=opts.keep_app,
        keep_env=opts.keep_env,
        grant_perm=opts.grant_perm,
        enable_accessibility_hard=opts.enable_accessibility_hard,
        ignore_ad=opts.ignore_ad)
    droidbot.start()

def main():
    """
    the main function
    it starts a droidbot according to the arguments given in cmd line
    """
    directory_path = "../step2/result_steps_text"
    file_list = sorted([f for f in os.listdir(directory_path) if f.endswith('.txt')])
    for filename in file_list:
        if filename != "reuters_TextSize.txt":
            continue
        extracted_info = process_files(directory_path, filename)
        for item in extracted_info:
            print(f"App: {item['app']}")
            print(f"Function: {item['function']}")
            print(f"Step Number: {item['step_number']}")
            print(f"Event/Assertion: {item['event_or_assertion']}")
            print(f"Task: {item['task']}")
            print(f"Status: {item['status']}\n")

        #explore(extracted_info)

if __name__ == "__main__":
    main()
