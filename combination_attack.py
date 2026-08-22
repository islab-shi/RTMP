# -*- coding: utf-8 -*-

from evolutionary_algorithms import EA
from json_opt import jsonOpt
from collections import Counter
import torch
import numpy as np
import random
import warnings
import copy
import math
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class combAttacker(EA):
    def __init__(self, model, networkName, popBound, numGenerations, popSize,
                 filterPosition, layerName, expectedAcc, x_test, y_test, batch_size, dnaSize=0, attack_config_list=[], device='cuda', className=[], onnxModel=None, testloader=None, json_file_path='attack_result'):
        EA.__init__(self, dnaSize, popBound, numGenerations, popSize, expectedAcc, popBound, model, x_test, batch_size, attack_config_list, onnxModel)
        self.device = device  # 'cpu' or 'cuda:0'
        self.model = model
        self.networkName = networkName
        self.filterPosition = filterPosition
        self.layerName = layerName
        self.x_test = x_test            # PyTorch path receives a tensor batch.
        self.y_test = y_test
        self.orgLayerWeight = self.get_layer_weight()
        self.jsonOpt = jsonOpt()
        self.className = className
        self.json_file_path = json_file_path

        # Added
        self.samplingPointNum = 200
        self.singleAttackRange = 9
        self.samplingPointPosition = []
        self.globalBestPredict = 1
        self.discrete = False
        self.batch_size = batch_size
        self.initialDistribution = self.get_topX_categories(len(self.className))
        self.initialLastFCValue = 0
        self.attack_config_list = attack_config_list
        self.sensitive_info_index = []
        self.sensitive_info_val   = []
        self.org_predit, self.org_top5_predit = self.predict_cal()

        # self.model.to(self.device)
        # self.model.eval()
        self.testloader = testloader
        self.current_predicted = None


    def set_atk_layer(self, layerName):
        self.layerName = layerName
        self.orgLayerWeight = self.get_layer_weight()

    def recover(self):
        # Restore the original PyTorch layer weights after a trial attack.
        for layer_idx in range(len(self.layerName)):
            layer_param = dict(self.model.named_parameters()).get(self.layerName[layer_idx] + '.weight')
            layer_param.data.copy_(self.orgLayerWeight[layer_idx])

    def sensitive_info_recover(self):
        self.sensitive_info_index = []
        self.sensitive_info_val = []

    def set_filterPosition(self, filterPosition):
        self.filterPosition = filterPosition

    def get_layer_weight(self):
        param_list = []
        # Copy the target layer weights before mutating them.
        try:
            for layer in self.layerName:
                layer_param = copy.deepcopy(self.model.state_dict()[layer + '.weight'])
                param_list.append(layer_param)
            return param_list
        except KeyError:
            raise ValueError(f"Layer {self.layerName} not found in the model.")




    def kernel_swap(self, dna, recover=True, advance_attack=False):
        layer_param_dict = {}

        if recover is True:
            self.recover()

        for layer_idx in range(len(self.layerName)):
            # get filter
            layer_param_dict[self.layerName[layer_idx]] = (dict(self.model.named_parameters()).get(self.layerName[layer_idx] + '.weight'))

        attack_config_num = len(self.attack_config_list)

        dna_start_pos = 0

        for c_idx in range(attack_config_num):
            attack_config = self.attack_config_list[c_idx]

            atk_layer_name = attack_config[4]
            attack_filters_num = attack_config[0]       # Number of consecutive filters modified by this attack.
            attack_kernels_num = attack_config[1]

            param = layer_param_dict[atk_layer_name]
            attack_weight_size = attack_config[5]
            weight_offset = attack_config[6]

            for f_idx in range(attack_filters_num):
                filter_pos = attack_config[2] + f_idx
                for k_idx in range(attack_kernels_num):
                    kernel_pos = attack_config[3] + k_idx

                    end_pos = dna_start_pos + attack_weight_size
                    # dna_value = np.array(dna[dna_start_pos : end_pos]).reshape(param.data.shape[2], param.data.shape[2])

                    # param.data[filter_pos][kernel_pos].copy_(torch.from_numpy(dna_value))

                    tmp_kernel = param.data[filter_pos][kernel_pos].reshape(param.data.shape[2]**2)
                    dna_value = torch.from_numpy(np.array(dna[dna_start_pos:end_pos]))
                    # dna_value = 0
                    # for i in range(len(dna_value)):
                    #     dna_value[i] = 0.33
                    tmp_kernel[weight_offset:attack_weight_size+weight_offset] = dna_value
                    param.data[filter_pos][kernel_pos].copy_(tmp_kernel.reshape(param.data.shape[2], param.data.shape[2]))

                    dna_start_pos += attack_weight_size


    def evaluate_get_topX_categories(self, topNum=10):
        result_list = []
        self.model.eval()  # Run evaluation with deterministic inference layers.

        with torch.no_grad():
            for data, _ in self.testloader:
                data = data.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                result_list.extend(predicted.tolist())

        catego = Counter(result_list).most_common(topNum)
        cateDict = {}
        class_list = self.className[:]

        for i in range(len(catego)):
            cateDict[self.className[catego[i][0]]] = catego[i][1]
            class_list.remove(self.className[catego[i][0]])

        for i in range(len(class_list)):
            cateDict[class_list[i]] = 0

        return cateDict

    def evaluate_topk_with_metrics(self):
        if self.networkName != 'YOLOv8':
            self.model.eval()  # Run evaluation with deterministic inference layers.

        all_preds = []
        all_targets = []


        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                if self.networkName == 'YOLOv8':
                    output = self.model.model(data)
                else:
                    output = self.model(data)

                _, pred = output.topk(5, 1, True, True)  # Get the top-5 predictions.

                # Store the Top-1 class for sklearn metrics.
                top1_pred = pred[:, 0]  # The first column stores the Top-1 prediction.
                all_preds.extend(top1_pred.cpu().tolist())  # Convert to a Python list for metric APIs.
                all_targets.extend(target.cpu().tolist())  # Convert targets to the same representation.

        # Calculate accuracy.
        accuracy = accuracy_score(all_targets, all_preds)

        # Build the confusion matrix.
        conf_matrix = confusion_matrix(all_targets, all_preds)

        # Build the classification report.
        class_report = classification_report(all_targets, all_preds)

        return accuracy, conf_matrix, class_report

    def evaluate_topk(self):
        self.model.eval()  # Run evaluation with deterministic inference layers.
        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, pred = output.topk(5, 1, True, True)  # Get the top-5 predictions.
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))

                top1_correct += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
                top5_correct += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
                total += target.size(0)

        top1_acc = top1_correct / total * 100
        top5_acc = top5_correct / total * 100

        return top1_acc, top5_acc

    def predict_cal(self, freeModel=False):
        if freeModel is True:
            self.recover()

        if self.networkName == 'YOLOv8':
            return 0, 0
        else:
            total = 0
            top1_correct = 0
            top5_correct = 0
            with torch.no_grad():
                # for data in self.x_test:
                #     images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(self.x_test)
                _, predicted = torch.max(outputs, 1)
                total += self.y_test.size(0)
                top1_correct += (predicted == self.y_test).sum().item()

                # Cache the latest inference result for later inspection.
                self.current_predicted = predicted

                _, predicted_top5 = torch.topk(outputs, 5, dim=1)
                top5_correct += sum(self.y_test[i].item() in predicted_top5[i] for i in range(self.y_test.size(0)))

            top1_accuracy = top1_correct / total
            top5_accuracy = top5_correct / total

            return top1_accuracy, top5_accuracy

    def get_lastFCValue(self):
        # pytorch
        catego_sum_list = []
        if self.networkName == 'YOLOv8':
            outputs = self.model(self.x_test, verbose=False)
            for result in outputs:
                catego_sum_list.append(result.probs.data)

            catego_sum_tensor = torch.stack(catego_sum_list).sum(dim=0)
            return catego_sum_tensor.cpu().numpy()
        else:
            softmax = torch.nn.Softmax(dim=1)
            with torch.no_grad():
                # for data in self.x_test:
                #     images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(self.x_test)
                outputs = softmax(outputs)
                for i in range(outputs.shape[0]):
                    catego_sum_list.append(outputs[i].clone())

            catego_sum_tensor = torch.stack(catego_sum_list).sum(dim=0)
            return catego_sum_tensor.cpu().numpy()

    def __attack(self, toleranceVal=0.003, toleranceRound=8, filterVal=0.02, recover=True, filePath="", advanceAttack=False):
        if self.discrete is False:
            print("Attack range: [" + str(self.filterPosition) + "--" + str(self.filterPosition + self.dnaSize) + ']')
        else:
            print("Attack range:",)
            for pos in self.filterPosition:
                print("["+str(pos) + "--" + str(pos + self.dnaSize//len(self.filterPosition)) + "] ")
            print("")

        self.create_pop()
        self.clear_bestFitness()
        self.clear_bestPredict()

        self.count = 0
        self.predictHistory = 0

        exit = False

        # roundNum: The number of actual runs
        for self.roundNum in range(self.numGenerations):
            print("gen " + str(self.roundNum) + ":")

            kids = self.make_kid()
            if advanceAttack is True:
                self.kill_bad_according_FCVal(kids)
            else:
                self.kill_bad(kids)

            deviation = {}
            tmpBestWeights = self.pop['DNA'][0].copy().tolist()
            self.kernel_swap(tmpBestWeights, advance_attack = advanceAttack)

            cateTopX = self.get_topX_categories(topNum=len(self.className))
            # print("cateTopX = " + str(cateTopX))
            cateTopX_print = {}
            for key, val in cateTopX.items():
                if val != 0:
                    cateTopX_print[key] = val
            print("cateTopX = " + str(cateTopX_print))

            for i in range(len(self.className)):
                difference = cateTopX[self.className[i]] - self.initialDistribution[self.className[i]]
                deviation[self.className[i]] = difference

            # lis = sorted(deviation.items(), key=lambda x: x[1], reverse=True)
            lis_rev = sorted(deviation.items(), key=lambda x: x[1], reverse=True)       # reverse=True sorts high to low.
            bestReduceCatgo = lis_rev[0][1]

            show_msg = []
            for i in range(len(self.className)):
                if lis_rev[i][1] != 0:
                    show_msg.append(lis_rev[i])

            print('result = ' + str(show_msg))

            if bestReduceCatgo == 0 and self.roundNum > 10:
                if recover is True:
                    self.model.params[self.layerName][0].data[...] = self.orgLayerWeight
                return False

            if advanceAttack is True:
                diff_list = []

                diff_list = self.get_lastFCValue() - self.initialLastFCValue #.append(self.get_lastFCValue()[self.attack_catego] - self.initialLastFCValue)                  # Increase a target class.

                print("max diff = " + str(max(diff_list)))
                print("max diff index = " + str(np.array(diff_list).argmax()))
                print("max diff catego = " + str(self.className[np.array(diff_list).argmax()]))
                if self.attack_catego == -1:
                    difference = max(diff_list)
                else:
                    difference = diff_list[self.attack_catego]

                if difference < 0.001 and self.roundNum >= 1 and self.attack_catego != -1:
                    exit = True

                if difference < 0.001 and self.roundNum > 30:
                    exit = True
                print("advance attack, difference val = " + str(difference))
                if self.roundNum == (self.numGenerations - 1):
                    diff_list_tmp = copy.copy(diff_list)
                    for i in range(100):
                        index = np.array(diff_list_tmp).argmax()
                        val = max(diff_list_tmp)
                        self.sensitive_info_index.append(index)
                        self.sensitive_info_val.append(val)
                        diff_list_tmp[index] = 0

                    if difference > 0.05 and self.roundNum > 5:
                        middle_name = ''
                        for atk_element in range(len(self.attack_config_list)):
                            attack_config = self.attack_config_list[atk_element]
                            middle_name += str(attack_config[0]) +' '+ str(attack_config[1]) +' '+ str(attack_config[2])
                            middle_name += '_'
                        self.__saveInfo(filePath + "_" +'atk_cfg' + self.className[self.attack_catego] + "_fcVal.json")
                        exit = True
            else:
                diff_predit = self.org_predit - self.predict_cal()
                if diff_predit > 0.05:
                    self.__saveInfo(filePath + "_" + str(self.popBound[1]) + ".json")
                    exit = True

            if exit is True:
                if recover is True:
                    self.recover()
                return

        if recover is True:
            self.recover()

        if exit is True:
            pass

    def export_to_quant_onnx(self):
        pass

    def get_layerInfo(self):
        # layerWeight.shape: (filterLength, filterWidth, IFM_channel, OFM_channel)
        filterSize = self.orgLayerWeight.shape[2]           # 5
        filterGroupSize = self.orgLayerWeight.shape[1]            # 32
        filterGroupNum = self.orgLayerWeight.shape[0]               # 64
        weightNum = (filterSize ** 2) * filterGroupSize * filterGroupNum
        return weightNum, filterSize, filterGroupSize, filterGroupNum

    def set_sampingPointNum(self, num):
        self.samplingPointNum = num

    def set_singleAttackRange(self, range):
        self.singleAttackRange = range

    def set_sampingPointPosition(self):
        # for i in range(self.samplingPointNum):
        self.samplingPointPosition = random.sample(range(0, 663552, self.dnaSize),
                                                   self.samplingPointNum)   # 36864... Fixed me

    def get_topX_categories(self, topNum=10):  # Demo helper retained from experiments; returns top predicted categories.
        result_list = []

        if self.networkName == 'YOLOv8':
            all_detected_classes = []
            with torch.no_grad():
                outputs = self.model(self.x_test, verbose=False)
                for result in outputs:
                    all_detected_classes.append(result.probs.top1)

            catego = Counter(all_detected_classes).most_common(topNum)
            cateDict = {}
            class_list = self.className[:]

            for i in range(len(catego)):
                cateDict[self.className[catego[i][0]]] = catego[i][1]
                class_list.remove(self.className[catego[i][0]])

            for i in range(len(class_list)):
                cateDict[class_list[i]] = 0

            return cateDict

        else:
            # pytorch
            with torch.no_grad():
                outputs = self.model(self.x_test)
                _, predicted = torch.max(outputs, 1)
                result_list.extend(predicted.tolist())

            catego = Counter(result_list).most_common(topNum)
            cateDict = {}
            class_list = self.className[:]

            for i in range(len(catego)):
                cateDict[self.className[catego[i][0]]] = catego[i][1]
                class_list.remove(self.className[catego[i][0]])

            for i in range(len(class_list)):
                cateDict[class_list[i]] = 0

            return cateDict


    def advanceAttack_2(self, round, attack_catego):
        filePath = f"{self.json_file_path}/advanceAttack_2_%s_%d-%d_atkNum_%d_eph_%d_" % (
            '_'.join(self.layerName),
            0,
            # self.filterPosition,
            # finalPosition,
            0,
            self.dnaSize,
            self.numGenerations)

        self.attack_catego = attack_catego
        self.initialLastFCValue = self.get_lastFCValue()

        position = self.filterPosition
        for i in range(round):
            self.set_filterPosition(position)
            self.__attack(filterVal=100, toleranceVal=0.003, toleranceRound=20, filePath=filePath, advanceAttack=True)  # Changed...

            position = self.dnaSize + self.filterPosition

    def __saveInfo(self, filePath):
        dumpData = {}
        dumpData['Range'] = [self.filterPosition, self.filterPosition + self.dnaSize] #'[' + str(self.filterPosition) + "-" + str(self.filterPosition + self.dnaSize) + ']'
        dumpData['RoundNum'] = self.roundNum + 1
        dumpData['bestAttackedWeight0'] = self.pop['DNA'][0].copy().tolist()
        dumpData['bestFitness'] = np.float64(self.get_bestFitness().copy())
        dumpData['bestPredict'] = np.float64(self.get_bestPredict().copy())

        self.jsonOpt.writeJson(dumpData, filePath)

        fcVal_list, _, _, _,_ = self.get_catego_val()

        for i in range(1):
            dumpData = {}
            dumpData['bestAttackedWeight'] = self.pop['DNA'][i].copy().tolist()
            dumpData['kernel_idx'] = self.attack_config_list
            self.jsonOpt.writeJson(dumpData, filePath)

    def attack(self, strategy, round=0, accurateStorage=False, samplingPointNum=1000, attack_catego=0):  # Strategy selects the attack search mode.
        self.roundNum = 0
        self.count = 0
        self.fitnessHistory = 0.

        if strategy == "advanceAttack_2":
            self.advanceAttack_2(round, attack_catego)
