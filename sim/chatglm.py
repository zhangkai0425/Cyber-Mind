import zhipuai
from VLM import VLM
import re
LLM_prompt = '''
你现在是一个机械臂任务的执行者，我会告诉你方块的名字和方块要放置的地点。请你帮我提取出方块名字和放置的坐标。
提示：
场地中每个方块都有名字，例如：cube1，cube2，cube3。。。。
有效的坐标范围在 0.1<x<0.7,0.1<y<0.7，如果坐标超出这个范围则输出[ERROR]
输出格式:请不要输出其他信息。
cube_name,x,y
下面是一个例子：
指令：请把 cube2 移动到 [0.3,0.3]，则输出应该是 cube2,0.3,0.3
请你只输出结果。
下面是我的指令：
'''
LLM_prompt_v2 = '''
你现在是一个机械臂的代理，需要将我的指令语言转化成对应的格式以供机械臂API调用。
我将询问你把方块移动到目标地点。
其中方块名字我会直接告诉你，例如 cube1,cube2,cube3...
目标地点则需要你去写一段文字去咨询视觉语言模型，由于视觉语言模型只能读英语，所以请你写一段英文去咨询视觉语言模型。
请你先识别我指令中的目的地，按照所给的模板：find the [object]。将你认为的目的地翻译成英文填入[object]中输出。
因此，你最终的输出将按照格式：[方块名字, 一段咨询视觉语言模型的英文字符串]。
下面，是我的指令语言：请严格按照格式输出内容，不要多余的信息。
'''
LLM_prompt_v3 = '''
你现在是一个机械臂的代理，需要将我的指令语言转化成对应的格式以供机械臂API调用。
我将询问你把方块移动到目标地点。
我现在已经告诉你了已有cube的信息
cube1 : 蓝色
cube2 : 红色
cube3 : 绿色
其中方块名字就是 cube1,cube2,cube3...,上面代码中有cube的颜色坐标之类属性的信息
目标地点则需要你去写一段文字去咨询视觉语言模型，由于视觉语言模型只能读英语，所以请你写一段英文去咨询视觉语言模型。
请你先识别我指令中的目的地，按照所给的模板：find the [object]。将你认为的目的地翻译成英文填入[object]中输出。
因此，你最终的输出将按照格式：[方块名字, 一段咨询视觉语言模型的英文字符串]。
下面，是我的指令语言：请严格按照格式输出内容，不要多余的信息。

'''
LLM_prompt_v4 = '''
你现在是一个机械臂的代理，需要将我的指令语言转化成对应的格式以供机械臂API调用。
我将询问你把方块移动到目标地点。
我现在已经告诉你了已有cube的信息
cube1 : 蓝色
cube2 : 红色
cube3 : 绿色
其中方块名字就是 cube1,cube2,cube3...,上面代码中有cube的颜色坐标之类属性的信息
目标地点则需要你去写一段文字去咨询视觉语言模型，由于视觉语言模型只能读英语，所以请你写一段英文去咨询视觉语言模型。
请你先识别我指令中的目的地，按照所给的模板：find the [object]。将你认为的目的地翻译成英文填入[object]中输出。
可以一步步推理方块名称和目的地，输出推理的过程
因此，你最终的输出将严格按照格式:$$方块名字,一段咨询视觉语言模型的英文字符串$$
中间一定要逗号隔开！
COT prompt:这是一个例子:
比如用户输入:"已知图片中有个红色的小鸟，我想把和小鸟颜色相同的方块放到黑色椅子的位置"
推理过程如下:
1.方块是和小鸟颜色相同，即红色方块，根据提供的方块信息，得知方块名字为cube2
2.目的地是黑色椅子，所以按照模版得到的指令是"find the black chair"
3.按照格式进行输出$$cube2,find the black chair$$
下面，是我的指令语言：请严格按照格式输出内容，不要多余的信息,不准在推理过程中或答案无关的文本中留下$$,以免我处理答案出现错误。
'''

LLM_prompt_v5 = '''
你现在是一个机械臂的代理，需要将我的指令语言转化成对应的格式以供机械臂API调用。
我将询问你把方块移动到目标地点。
我现在已经告诉你了已有cube的信息
cube1 : 蓝色
cube2 : 红色
cube3 : 绿色
其中方块名字就是 cube1,cube2,cube3...,上面代码中有cube的颜色坐标之类属性的信息
目标地点则需要你去写一段文字去咨询视觉语言模型，由于视觉语言模型只能读英语，所以请你写一段英文去咨询视觉语言模型。
请你先识别我指令中的目的地，按照所给的模板：find the [object]。将你认为的目的地翻译成英文填入[object]中输出。
因此，你最终的输出将严格按照格式:$$方块名字,一段咨询视觉语言模型的英文字符串$$
'''

warning_prompt = '''
你的格式不对，请仔细检查格式，不要输出任何多余的信息!
仔细检查目标位置是否是用户描述的目标!
重新输出:$$方块名字,一段咨询视觉语言模型的英文字符串$$
其中$$是标志符，能够让我快速找到输出
类似如下的输出格式是合法的:
$$cube1,"find the blue cube"$$
不要输出任何多余的信息，不需要道歉之类的话，只需要输出结果!
'''

class ChatGLM:
    def __init__(self, api_key) -> None:
        zhipuai.api_key = api_key
        self.history = []
        self.max_history_pairs = 10
        self.prompt = ""
        self.vlm = VLM()
        self.image_path = "../background_3.jpg"
    def read_prompt(self, txt_path:str):
        with open(txt_path, 'r', encoding='utf-8') as f:
            self.prompt = f.read()
        return self.prompt
    def set_prompt(self, new_prompt):
        self.prompt = new_prompt
    def get_prompt(self):
        return self.prompt
    def Chat(self, user_query):
        # Add the current user prompt to the history
        # prompt = self.history + [{"role": "user", "content": user_prompt}]
        user_prompt = LLM_prompt_v3 + user_query
        response = zhipuai.model_api.invoke(
            model="chatglm_turbo",
            prompt=[{"role": "user", "content": user_prompt}],
            top_p=0.3,
            temperature=0,
        )
        if response['code']!=200:
            print(f"GLM: connect error ({response['code']}))")
        else:
            out = response['data']['choices'][0]['content'][3:-2]
        name, vlm_prompt = out.split(',')
        print("out = ",out)
        print(name, vlm_prompt)
        [x,y] = self.vlm.find_object_pos(prompt=vlm_prompt,image_path=self.image_path)
        # print(f'target_pos: {target_pos}')
        return name, [x,y,0.05]
    def Robot_Instruction_Checker(self, out):
        name, vlm_prompt = out.split(',')
        # Rule 1
        if name not in ["cube1","cube2","cube3"]:
            print("name = ",name)
            return False
        # Rule 2: TODO
        return True
    def Chat_With_History(self, user_prompt):
        # Add the current user prompt to the history
        prompt = self.history + [{"role": "user", "content": user_prompt}]

        response = zhipuai.model_api.sse_invoke(
            model="chatglm_turbo",
            prompt=prompt,
            temperature=0,
            top_p=0.3,
            # incremental=True,
        )
        self.answer = ""
        self.history.append({"role": "user", "content": user_prompt})
        for event in response.events():
            if event.event == "add":
                self.answer += event.data
            elif event.event == "error" or event.event == "interrupted":
                print("Error")
            elif event.event == "finish":
                self.history.append(
                    {"role": "assistant", "content": self.answer})
            else:
                self.answer += event.data

        # Limit history to the specified number of pairs
        self.history = self.history[-(self.max_history_pairs * 2):]

        return self.answer
    
    def Decision(self, user_query):
        # Chat, Feedback and Decision
        out = self.Chat_With_History(LLM_prompt_v5 + user_query)
        print("Initial Answer = ", out)
        match = re.search(r'\$\$(.*?)\$\$', out)
        if match:
            out = match.group(1)
        else :
            out = "Matching failed, return"
        print("Filtered Out = ",out)
        name, vlm_prompt = out.split(',')
        print(name, vlm_prompt)
        while (self.Robot_Instruction_Checker(out) != True):
            print("Format wrong:Multiple rounds of dialogue feedback...")
            out = self.Chat_With_History(warning_prompt)
            print("Initial Answer = ", out)
            match = re.search(r'\$\$(.*?)\$\$', out)
            if match:
                out = match.group(1)
            else :
                out = "Matching failed, return"
            print("Filtered Out = ", out)
            name, vlm_prompt = out.split(',')
            print(name, vlm_prompt)
        
        print("Format Correct:Starting the VLM model")
        [x, y] = self.vlm.find_object_pos(prompt=vlm_prompt, image_path=self.image_path)
        return name, [x, y, 0.05]

if __name__ == "__main__":
    # Example usage:
    api_key = "19fa2ef4dd48412aa45a75dde70d3a21.sJZ7dp76fRRjRDWQ"
    glm = ChatGLM(api_key)

    # for i in range(10):
    #     my_promt = input(">>>")
    #     answer = ChatGLM.Chat(my_promt, max_history_pairs=10)
    #     print(answer)
    # prompt = glm.read_prompt(r'D:\Nvidia-Omniverse\pkg\isaac_sim-2023.1.0-hotfix.1\WorkSpace\AML\sim\prompt.txt')
    # query = "请把绿色方块移动到椅子处"
    query = input("请输入要求：")
    # print(prompt)
    name, [x,y,z] = glm.Decision(query)
    print(name)
    print(x)
    print(y)
    print(z)
    # 请将红色方块放置在绿色骨头位置
