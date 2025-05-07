from infra import AndroidController, UIHierarchy, Element, center, parse_bound, MainEvaluator
from infra import ActionType, Action, click_action, longclick_action, text_action, swipe_action, enter_action, back_action, stop_action
from config import apk_info
from pathlib import Path
import subprocess
import json
import cv2
import time
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

def get_connected_devices():
    # 执行adb devices命令
    output = subprocess.check_output(['adb', 'devices']).decode('utf-8')
    
    # 分割输出文本，过滤掉标题行和空行，获取设备列表
    devices = [line.split('\t')[0] for line in output.splitlines()[1:] if line.strip()]
    
    return devices

def get_only_device():
    # 获取当前连接的设备
    devices = get_connected_devices()
    
    # 如果没有设备连接，返回None
    if not devices:
        return None
    
    # 如果有多个设备连接，返回None
    if len(devices) > 1:
        return None
    
    return devices[0]

def bounds_step(ui_hierarchy: UIHierarchy) -> Tuple[Element | None, str]:
    print("\nAction selected, type 'MISTAKE' and press Enter if the wrong action has been selected")
    bounds = input("Enter element bounds:\n")
    if bounds == "MISTAKE":
        return None, bounds
    element = ui_hierarchy.find_element({"bounds": bounds})
    if element is None:
        print("ELEMENT NOT FOUND, please try again.")
        return bounds_step(ui_hierarchy)
    return element, bounds

def action_step(control, ui_hierarchy: UIHierarchy) -> dict:
    '''
    Receive user input and perform the specified action of the user
    Returns the action details of the user
    '''
    action = dict()
    action["type"] = input("\nEnter action type (click, longclick, input, swipe, enter, back, stop):\n").lower()
    element = None
    match action["type"]:
        case "click":
            element, bounds = bounds_step(ui_hierarchy)
            if element is None:
                return action_step(control, ui_hierarchy)
            control.click(*center(parse_bound(bounds)))
        case "longclick":
            element,bounds = bounds_step(ui_hierarchy)
            if element is None:
                return action_step(control, ui_hierarchy)
            control.tap_hold(*center(parse_bound(bounds)), 2)
        case "input":
            element, bounds = bounds_step(ui_hierarchy)
            if element is None:
                return action_step(control, ui_hierarchy)
            text = input("\nEnter input text:\n")
            action["message"] = text
            control.click(*center(parse_bound(bounds)))
            time.sleep(1)
            control.input(text)
        case "swipe":
            element, bounds = bounds_step(ui_hierarchy)
            if element is None:
                return action_step(control, ui_hierarchy)
            print("\nswipe in your emulator since we don't have the from coordinate and to coordinate.")
        case "enter":
            control.enter()
        case "back":
            control.back()
        case "stop":
            pass
        case _:
            print("\nInvalid action type. Please reenter the action.")
            return action_step(control, ui_hierarchy)
    action["element"] = element._attrib if element is not None else None
    if action["type"] == "stop":
        pass
    return action

def CollectTrace(trace_dir: Path):
    if (trace_dir / "meta.json").exists():
        print("\nTrace already exists. Skip collecting trace.")
    else:
        index = 0

        print("=" * 20)
        print("Start collecting the trace.\n")

        if MODE == "PRE":
            meta_data = []
            activities = []
            while True:
                input("\nPress Enter to capture screen and dump hierarchy.")
                
                screenshot = controller.capture_screen()
                hierarchy_str = controller.dumpstr()
                ui_hierarchy = controller.dump()
                activity = controller.activity()
                activities.append(activity.info())

                cv2.imwrite(str(trace_dir / f"{index}.png"), screenshot)
                with open(trace_dir / f"{index}.xml", 'w', encoding="utf-8") as f:
                    f.write(hierarchy_str)

                print("Screenshot and hierarchy saved.")

                action = action_step(controller, ui_hierarchy)
                meta_data.append(action)
                index += 1
                if action["type"] == "stop":
                    break 
            with open(trace_dir / "meta.json", 'w', encoding="utf-8") as f:
                json.dump(meta_data, f, indent=4)
            with open(trace_dir / "activities.json", 'w', encoding="utf-8") as f:
                json.dump(activities, f, indent=4)

        else:
            # maybe first collect the whole trace and then select which to save?
            raise NotImplementedError

def TranslateToAction(raw: dict) -> Action:
    match raw["type"]:
        case "click":
            return click_action(coord = center(parse_bound(raw["element"]["bounds"])))
        case "longclick":
            return longclick_action(coord = center(parse_bound(raw["element"]["bounds"])))
        case "input":
            return text_action(raw["message"], coord=center(parse_bound(raw["element"]["bounds"])))
        case "swipe":
            return swipe_action(coord_from = center(parse_bound(raw["element"]["bounds"])), coord_to = center(parse_bound(raw["element"]["bounds"])))
        case "enter":
            return enter_action()
        case "back":
            return back_action()
        case "stop":
            return stop_action()
        case _:
            raise ValueError("Invalid action type.")

def EvaluateTrace(trace_dir: Path) -> bool:
    # check whether the evaluator and trace is valid(or matched)

    print("\nStart evaluating the trace based on given evaluators.\n")
    evaluator = MainEvaluator(trace_dir / "evaluator.json")
    with open(trace_dir / "meta.json") as f:
        meta_data = json.load(f)
    with open(trace_dir / "activities.json") as f:
        activities = json.load(f)
    activities = list(map(tuple, activities))
    hierarchies = []
    for i in range(len(meta_data)):
        with open(trace_dir / f"{i}.xml", 'r', encoding="utf-8") as f:
            xml = f.read()
        hierarchies.append(UIHierarchy(ET.fromstring(xml)))
    actions = [TranslateToAction(raw) for raw in meta_data]
    result = evaluator.evaluate(hierarchies, actions, activities)

    if not result:
        print("\nEvaluator failed. Check your evaluator or trace.")
        print("If you want to re-collect the trace, delete the trace folder directly and rerun this script.")
        print("If you want to re-collect the evaluator, delete the evaluator.json file in the trace folder and rerun this script.")
        exit(1)
    else:
        print("Evaluation success, continuing to skills")



def PromptType(t: str):
    if t not in ["match", "check", "action_match", "element_match"]:
        raise ValueError("Invalid prompt type")
    response = input(f"\nEnter {t} type (equal, include):\n").lower()
    if response == "":
        response = "equal"
    if response not in ["equal", "include"]:
        print(f"Invalid {t} type, please try again")
        return PromptType(t)
    return response

def CollectEvaluator(trace_dir: Path):
    if (trace_dir / "evaluator.json").exists():
        print("\nEvaluator already exists. Skip collecting evaluator.")
    else:
        print("=" * 20)
        print("Start giving trace evaluation method.\n")

        evaluators = []
        while True:
            print("\nIf no more evaluation method, directly press Enter to exit.")
            evaluator = dict()
            evaluator["type"] = input("Enter evaluation type (stoppage, findelement, lastaction, findaction, findelementbyaction):\n").lower()
            match evaluator["type"]:
                case "stoppage":
                    evaluator["match_type"] = PromptType("match")
                    match_rules = input("\nEnter match rules with format (key1:value1;key2:value2;...):\n(the key must be element attribute(e.g. resource-id, content-desc, text)):\n").split(";")
                    match_rules = {rule.split(":")[0]: rule.split(":",1)[1] for rule in match_rules}
                    evaluator["match_rules"] = match_rules

                    evaluator["check_type"] = PromptType("check")
                    print("\nIf no check rules(e.g. you just need check existence of a element), directly press Enter.\n")
                    check_rules = input("Enter check rules with format (key1:value1;key2:value2;...)\n(the key must be element attribute(e.g. resource-id, content-desc, text) or (activity)):\n")
                    check_rules = {} if check_rules == "" else check_rules.split(";") 
                    evaluator["check_rules"] = {rule.split(":")[0]: rule.split(":",1)[1] for rule in check_rules}
                
                case "lastaction":
                    evaluator["check_type"] = PromptType("check")
                    check_rules = input("\nEnter check rules with format (key1:value1;key2:value2;...)\n(the key must be element attribute(e.g. resource-id, content-desc, text) or action attribute(e.g. action_type, message)):\n").split(";")
                    evaluator["check_rules"] = {rule.split(":")[0]: rule.split(":",1)[1] for rule in check_rules}
                
                case "findaction":
                    evaluator["match_type"] = PromptType("match")
                    match_rules = input("\nEnter match rules with format (key1:value1;key2:value2;...)\n(the key must be element attribute(e.g. resource-id, content-desc, text) or action attribute(e.g. action_type, message)):\n").split(";")
                    evaluator["match_rules"] = {rule.split(":")[0]: rule.split(":",1)[1] for rule in match_rules}

                    evaluator["check_type"] = PromptType("check")
                    print("\nIf no check rules(e.g. you just need check existence of a action), directly press Enter.\n")
                    check_rules = input("Enter check rules with format (key1:value1;key2:value2;...)\n(the key must be element attribute(e.g. resource-id, content-desc, text) or action attribute(e.g. action_type, message)):\n")
                    check_rules = {} if check_rules == "" else check_rules.split(";") 
                    evaluator["check_rules"] = {rule.split(":")[0]: rule.split(":",1)[1] for rule in check_rules}

                case "findelement":
                    evaluator["match_type"] = PromptType("match")
                    match_rules = input("\nEnter match rules with format (key1:value1;key2:value2;...):\n(the key must be element attribute(e.g. resource-id, content-desc, text) or (activity)):\n").split(";")
                    evaluator["match_rules"] = {rule.split(":")[0]: rule.split(":",1)[1] for rule in match_rules}

                    evaluator["check_type"] = PromptType("check")
                    print("\nIf no check rules(e.g. you just need check existence of a element), directly press Enter.\n")
                    check_rules = input("Enter check rules with format (key1:value1;key2:value2;...)\n(the key must be element attribute(e.g. resource-id, content-desc, text) or (activity)):\n")
                    check_rules = {} if check_rules == "" else check_rules.split(";") 
                    evaluator["check_rules"] = {rule.split(":")[0]: rule.split(":",1)[1] for rule in check_rules}

                case "findelementbyaction":
                    evaluator["action_match_type"] = PromptType("action_match")
                    action_match_rules = input("\nEnter action match rules with format (key1:value1;key2:value2;...)\n(the key must be element attribute(e.g. resource-id, content-desc, text) or action attribute(e.g. action_type, message)):\n").split(";")
                    evaluator["action_match_rules"] = {rule.split(":")[0]: rule.split(":",1)[1] for rule in action_match_rules}

                    evaluator["element_match_type"] = PromptType("element_match")
                    element_match_rules = input("\nEnter element match rules with format (key1:value1;key2:value2;...)\n(the key must be element attribute(e.g. resource-id, content-desc, text)):\n").split(";")
                    evaluator["element_match_rules"] = {rule.split(":")[0]: rule.split(":",1)[1] for rule in element_match_rules}
                    
                    evaluator["check_type"] = PromptType("check")
                    print("\nIf no check rules(e.g. you just need check existence of a element), directly press Enter.\n")
                    check_rules = input("Enter check rules with format (key1:value1;key2:value2;...)\n(the key must be element attribute(e.g. resource-id, content-desc, text)):\n")
                    check_rules = {} if check_rules == "" else check_rules.split(";")
                    evaluator["check_rules"] = {rule.split(":")[0]: rule.split(":",1)[1] for rule in check_rules}

                case "":
                    break

                case _:
                    print("\nInvalid evaluation type, please try again")
                    evaluator = None
            if evaluator != None:
                evaluators.append(evaluator)

        # dump the evaluator
        with open(trace_dir / "evaluator.json", 'w', encoding="utf-8") as f:
            json.dump(evaluators, f, indent=4)
    
def CollectSkill(trace_dir: Path):
    if (trace_dir / "skills.json").exists():
        print("\nSkills already exists. Skip collecting skills.")
    else:
        print("=" * 20)
        print("Start collecting skills\n")
        with open(trace_dir / "meta.json") as f:
            meta_data = json.load(f)
        length = len(meta_data) - 1
        start = 0
        skills = []
        while start < length:

            print("\n\n")
            for i in range(start, length):
                element = meta_data[i]["element"]
                message = f"index-{i - start + 1} ===== type: {meta_data[i]['type']}"
                if meta_data[i]['type'] == "input":
                    message += f", text: {meta_data[i]['message']}"
                if element is not None:
                    message += ", element: { "
                    resource_id = element["resource-id"]
                    if resource_id != "":
                        message += f"resource-id: {resource_id} , "
                    content_desc = element["content-desc"]
                    if content_desc != "":
                        message += f"content-desc: {content_desc} , "
                    text = element["text"]
                    if text != "":
                        message += f"text: {text} , "
                    message += "}"
                print(message)

            skill_len = int(input("\nPlease enter the length of the skill: \n"))
            skill_name = input("\nPlease enter the name of the skill: \n")
            skill = {"name": skill_name, "length": skill_len, "start_index": start, "end_index": start + skill_len - 1}
            skills.append(skill)
            start += skill_len
            assert start <= length
        with open(trace_dir / "skills.json", "w", encoding="utf-8") as f:
            json.dump(skills, f, indent=4)


MODE = "PRE"
# MODE = "POST"

if __name__ == "__main__":

    with open("task_info.json") as f:
        task_info = json.load(f)

    port = get_only_device()

    if port is None:
        print("\nUse 'adb devices' to check connected devices.\n")
        port = input("Enter emulator port (example: emulator-5554):\n")

    task_id = int(input("\nEnter task id:\n"))
    apk_name = input("\nEnter apk name (do not include '.apk'):\n")

    task = [t for t in task_info if t['id'] == task_id][0]
    assert apk_name in task['apps']
    print("\nTask info: ", task)

    trace_dir = Path("groundtruth") / str(task_id) / apk_name
    trace_dir.mkdir(parents=True, exist_ok=True)

    apk_path = apk_info[apk_name]['path']
    package = apk_info[apk_name]['package']
    username = apk_info[apk_name]['username']
    password = apk_info[apk_name]['password']

    controller = AndroidController(port, package)

    # reinstall and start    or    stop and start?
    

    # controller.reinstall_app(apk_name, apk_path)
    # controller.start_app()

    controller.stop_app()
    controller.start_app()

    # maybe script login?
    # login

    CollectTrace(trace_dir)

    CollectEvaluator(trace_dir)    

    EvaluateTrace(trace_dir)

    CollectSkill(trace_dir)
    
    controller.stop_app()