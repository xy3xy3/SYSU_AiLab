class StuData:
    def __init__(self, file:str):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                tmp = f.readlines()
                self.data = []
                for i in tmp:
                    self.data.append(i.strip().split())
        except FileNotFoundError:
            print(file,'文件不存在')
            self.data = []  # 如果文件不存在，将self.data初始化为空列表

    def AddData(self, name="Bob", stu_num="003", gender="M", age=20):
        self.data.append([name, stu_num, gender, age])

    def SortData(self, type:str):
        if type == 'name':
            self.data.sort(key=lambda x: x[0])
        elif type == 'stu_num':
            self.data.sort(key=lambda x: x[1])
        elif type == 'gender':
            self.data.sort(key=lambda x: x[2])
        elif type == 'age':
            self.data.sort(key=lambda x: int(x[3]))

    def ExportFile(self, file: str):
        with open(file, 'w') as f:
            for i in self.data:
                f.write(' '.join(map(str, i)) + '\n')

    def PrintData(self):
        if hasattr(self, 'data'):  # 检查是否存在 self.data 属性
            print(self.data)
        else:
            print("没有学生数据可供打印")

if __name__ == "__main__":
    sd = StuData(r'c:\BaiduSyncdisk\AiLab\week2\Code\student_data.txt')
    sd.PrintData()
    sd.AddData('Tomb', '004', 'M', 21)
    sd.PrintData()
    sd.SortData('age')
    sd.PrintData()
    sd.ExportFile(r'c:\BaiduSyncdisk\AiLab\week2\Code\student_data2.txt')
