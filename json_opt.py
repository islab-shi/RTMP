import json
import os
import re

class jsonOpt:
    def __init__(self):
        self.cnt = {}
        self.jsonData = {}

    def clearCnt(self):
        self.cnt = {}

    def readJson(self, filename):
        file = open(filename, 'r')
        self.jsonData = json.load(file)
        index = list(self.jsonData)[-1]
        self.cnt[filename] = int(re.findall(r"\d+\.?\d*", index)[0]) + 1
        file.close()

    def writeJson(self, dataDict, filename='dump'):
        if os.path.exists(filename) and filename not in self.cnt:
            self.readJson(filename)
        elif filename not in self.cnt:
            self.cnt[filename] = 0

        key = 'member_' + str(self.cnt[filename])
        jsonData = {}
        jsonData[key] = dataDict

        self.__json_append(filename, jsonData)
        self.cnt[filename] += 1

    def __json_append(self, filename, dic):
        try:
            if os.path.exists(filename):
                dic_json = json.dumps(dic, indent=4)
                ch1 = ", "
                ch2 = "}"
                dic1 = ch1 + dic_json[1:(len(dic_json) - 1)] + ch2
                try:
                    with open(filename, "rb+") as f:
                        f.seek(-1, 2)
                        f.write(bytes(dic1, encoding='utf-8')) # f.write(str(dic1)) In python2     # f.write(bytes(dic1, encoding='utf-8')) In python3
                        return 0
                except IOError:
                    print("Write %s failed" % filename)
                    return -1
            else:
                try:
                    with open(filename, "w") as f:
                        json.dump(dic, f, indent=4)
                        return 0
                except IOError:
                    print("Write %s failed" % filename)
                    return -1
        except IOError:
            print("Call the io module to find %s failed" % filename)
            return -2