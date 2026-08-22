# -*- coding: utf-8 -*-
import numpy as np
import math
import torch


class EA():
    def __init__(self, dnaSize, popBound, numGenerations, popSize, expectedAcc, attack_maxVal, model, x_test, batch_size, attack_config_list=[], onnxModel=None):
        self.dnaSize = dnaSize                      # Number of weights encoded by one DNA vector.
        self.popBound = popBound                    # Default attack weight ranges.
        self.numGenerations = numGenerations        # Number of evolution generations.
        self.popSize = popSize                      # Initial population size.
        self.numKid = round(popSize / 2)
        self.expectedAcc = expectedAcc
        self.best_fitness = []
        self.best_predict = []
        self.roundNum = 0                           # Number of generations actually executed.
        self.pop = dict()
        self.model = model
        self.x_test = x_test
        self.batch_size = batch_size
        self.attack_maxVal = [bound[1] for bound in self.popBound]
        self.attack_catego = 0
        self.attack_layer_dna_size = attack_config_list
        self.onnxModel = onnxModel
        self.org_quantParam = []
        self.quant_scales = []
        self.bn_scales = []
        self.attack_layer_idx_list = []
        self.use_hamming_dist = False
        # self.initialLastFCValue = self.get_lastFCValue()

    # @abc.abstractmethod
    def kernel_swap(self, dna, recover=True, advance_attack=False):
        pass

    # @abc.abstractmethod
    def predict_cal(self, freeModel=False):             # freeModel=True evaluates the recovered clean model.
        pass

    def set_popSize(self, popSize):
        self.popSize = popSize
        self.numKid = round(popSize / 2)

    def get_lastFCValue(self):
        pass

    # @abc.abstractmethod
    # def __attack(self, toleranceVal=0.01, toleranceRound=6, filterVal=0.12):
    #     pass

    # def set_filterPosition(self, filterPosition):
    #     self.filterPosition = filterPosition

    def set_dnaSize(self, dnaSize):
        self.dnaSize = dnaSize

    def get_bestFitness(self):
        return self.best_fitness[-1]

    def get_bestPredict(self):
        return self.best_predict[-1]

    def clear_bestFitness(self):
        self.best_fitness = []

    def clear_bestPredict(self):
        self.best_predict = []

    def get_topX_categories(self, topNum=10):
        pass

    def create_pop(self):
        self.pop = dict(DNA=np.empty((self.popSize, self.dnaSize)),
                        mut_strength=np.random.rand(self.popSize, self.dnaSize))

        # Initialize DNA values by layer-specific bounds.
        start = 0
        next_attack_maxVal_idx = 0
        last_layer = 0
        for i in range(len(self.attack_layer_dna_size)):
            for j in range(self.attack_layer_dna_size[i][1]):
                if i == 0:
                    last_layer = self.attack_layer_dna_size[i][4]

                end = start + self.attack_layer_dna_size[i][5]

                if self.attack_layer_dna_size[i][4] == last_layer:
                    self.pop['DNA'][:, start:end] = (2 * np.random.rand(self.popSize, end - start) - 1) * self.attack_maxVal[next_attack_maxVal_idx]
                else:
                    last_layer = self.attack_layer_dna_size[i][4]
                    next_attack_maxVal_idx += 1
                    self.pop['DNA'][:, start:end] = (2 * np.random.rand(self.popSize, end - start) - 1) *  self.attack_maxVal[next_attack_maxVal_idx]

                start = start + self.attack_layer_dna_size[i][5]

            # Force the first individual to the maximum bound for each layer.
        start = 0
        next_attack_maxVal_idx = 0
        last_layer = 0
        for i in range(len(self.attack_layer_dna_size)):
            for j in range(self.attack_layer_dna_size[i][1]):
                if i == 0:
                    last_layer = self.attack_layer_dna_size[i][4]

                end = start + self.attack_layer_dna_size[i][5]

                if self.attack_layer_dna_size[i][4] == last_layer:
                    self.pop['DNA'][0, start:end] = np.ones(end - start) * self.attack_maxVal[next_attack_maxVal_idx]
                else:
                    last_layer = self.attack_layer_dna_size[i][4]
                    next_attack_maxVal_idx += 1
                    self.pop['DNA'][0, start:end] = np.ones(end - start) * self.attack_maxVal[next_attack_maxVal_idx]

                start = start + self.attack_layer_dna_size[i][5]

    # Apply each DNA vector to the attacked layer and collect predictions.
    def get_predict(self):
        predict = []
        dnas = self.pop['DNA']

        for dna in dnas:  # dnsa_size = N_KID + POP_SIZE
            self.filter_swap(dna)
            accuracyCifar10 = self.predict_cal()
            predict.append(accuracyCifar10)

        return np.array(predict)

    def get_fitness(self, predict):
        for idx in range(len(predict)):
            predict[idx] = abs(predict[idx] - self.expectedAcc)
        return predict

    def kill_bad(self, kids):
        # Merge the parent population with generated children.
        for key in ['DNA', 'mut_strength']:
            self.pop[key] = np.vstack((self.pop[key], kids[key]))
        predict = self.get_predict()
        predict_cpy = predict.copy()  # Keep the raw predictions for best_predict.

        fitness = self.get_fitness(predict_cpy)
        idx = np.arange(self.pop['DNA'].shape[0])
        good_idx = idx[fitness.argsort()][:self.popSize]

        for key in ['DNA', 'mut_strength']:
            self.pop[key] = self.pop[key][good_idx]

        # print("best Deviation = " + str(fitness[good_idx[0]])[0:7], str(fitness[good_idx[1]])[0:7],
        #       str(fitness[good_idx[2]])[0:7])  # Keep a compact numeric display.
        self.best_fitness.append(fitness[good_idx[0]])
        self.best_predict.append(predict[good_idx[0]])
        print("predict: " + str(self.best_predict[-1]))

    def get_fitness_catego(self, fcVal, total_hamming_distance_list, total_overflow_number_list, dist):
        if self.use_hamming_dist is True:
            for idx in range(len(fcVal)):
                fc = abs(fcVal[idx] - len(self.x_test)*1) * 2
                # Allow a small tolerance before adding distance penalties.
                # if fc <= 6:
                #     fc = 0
                overflow_num = total_overflow_number_list[idx] * 15
                hamming_distance = total_hamming_distance_list[idx]
                # non_zero_count = sum(1 for x in dist[idx] if x != 0)
                punish_overflow = overflow_num
                punish_hamming_distance = hamming_distance * 6            # 1.2 for vgg, 3.7 for Resnet-18

                if hamming_distance <= 31:
                    punish_hamming_distance = 0

                fcVal[idx] = fc + punish_overflow + punish_hamming_distance
        else:
            for idx in range(len(fcVal)):
                fcVal[idx] = abs(fcVal[idx] - len(self.x_test) )          # TODO: Revisit whether this should match the input batch size.

        return fcVal
    def set_attack_layer_idx_list(self, attack_layer_idx_list):
        self.attack_layer_idx_list = attack_layer_idx_list
        self.__get_bn_and_quant_param()

    def __get_bn_and_quant_param(self):
        import onnx
        from ultralytics import YOLO

        def get_node_info(model, node):
            weights = {}
            for input_name in node.input:
                for initializer in model.graph.initializer:
                    if initializer.name == input_name:
                        weights[input_name] = onnx.numpy_helper.to_array(initializer)
            return weights

        def find_node_by_name(model, node_name):
            for node in model.graph.node:
                if node.name == node_name:
                    return get_node_info(model, node)
            return None


        # attack_layer_name_onnx = ['/layer2/layer2.1/conv2/Conv_quant', '/layer2/layer2.1/conv1/Conv_quant',
        #                           '/layer2/layer2.0/downsample/downsample.0/Conv_quant', '/layer2/layer2.0/conv2/Conv_quant'] # attack_layer_idx_list = [32, 29, 26, 23]

        attack_layer_name_onnx = ['/layer2/layer2.0/conv1/Conv_quant', '/layer1/layer1.1/conv2/Conv_quant', '/layer1/layer1.1/conv1/Conv_quant']  # [20, 16, 13]
        attack_layer_name_onnx = ['/model.4/m.3/cv1/conv/Conv_quant', '/model.4/m.2/cv2/conv/Conv_quant', '/model.4/m.2/cv1/conv/Conv_quant']
        attack_layer_name_onnx = ['/model.6/m.3/cv2/conv/Conv_quant', '/model.6/m.3/cv1/conv/Conv_quant',
                                  '/model.6/m.2/cv2/conv/Conv_quant']

        # attack_layer_name_onnx = ['/features/features.17/Conv_quant', '/features/features.14/Conv_quant', '/features/features.12/Conv_quant', '/features/features.10/Conv_quant']


        tmp_model = YOLO("yolov8m-cls.pt")
        modules = dict(tmp_model.named_modules())
        modules = list(modules.items())

        # BatchNorm is expected to immediately follow Conv in the exported graph.
        bn_layers = []
        for conv_idx in self.attack_layer_idx_list:
            bn_idx = conv_idx + 1
            bn_layers.append(modules[bn_idx][1].state_dict())

        layer_idx = 0
        current_layer = self.attack_layer_dna_size[0][4]
        for attack_config in self.attack_layer_dna_size:
            target_layer_onnx_list = []
            for layer_onnx_name in attack_layer_name_onnx:
                target_layer_onnx_list.append(list(find_node_by_name(self.onnxModel, layer_onnx_name).items()))

            if attack_config[4] != current_layer:
                layer_idx = layer_idx + 1
                current_layer = attack_config[4]

            filter_idx = attack_config[2]
            kernel_idx = attack_config[3]
            elem_idx = attack_config[-1]

            # VGG-16 does not include BatchNorm layers.
            if not bn_layers[layer_idx] or 'YOLOv8' in self.networkName:
                self.bn_scales.append(torch.tensor(1))
                bn_scale = torch.tensor(1)
            else:
                bn_params = bn_layers[layer_idx]
                bn_scale_1 = (1 / math.sqrt(bn_params['running_var'][filter_idx] - 0.0000099))
                bn_scale_2 = bn_params['weight'][filter_idx]
                bn_scale = bn_scale_1 * bn_scale_2
                self.bn_scales.append(bn_scale)

            quant_scale = target_layer_onnx_list[layer_idx][3][1]
            self.quant_scales.append(quant_scale)

            if 'YOLOv8' not in self.networkName:
                self.popBound[layer_idx][1] = 127 / bn_scale.cpu().tolist() * quant_scale

            # Read the quantized parameter for the attacked layer.
            quant_weight = target_layer_onnx_list[layer_idx][2][1]     # Quantized weights.
            target_quant_weight = quant_weight[filter_idx][kernel_idx].flatten()[elem_idx]
            self.org_quantParam.append(target_quant_weight)

    def hamming_distance_int8(self, a, b):
        # Treat both operands as int8 values.
        a &= 0xFF
        b &= 0xFF

        # XOR highlights the changed bit positions.
        xor_result = a ^ b
        # Count the number of changed bits.
        hamming_distance = bin(xor_result).count('1')

        return hamming_distance

    def get_catego_val(self):
        def round_tensor(tensor):
            return tensor.round().int().tolist()

        val_list = []
        dnas = self.pop['DNA']
        total_hamming_distance_list = []
        total_overflow_number_list = []
        tmp = []
        total_org_quant_qp_list = []
        for dna in dnas:  # dnsa_size = N_KID + POP_SIZE
            # Optionally include the Hamming-distance penalty.
            if self.use_hamming_dist is True:
                total_hamming_distance = 0
                overflow_num = 0
                dist_list = []
                qp_list = []
                for idx in range(len(dna)):
                    bn_scale = self.bn_scales[idx]
                    quant_scale = self.quant_scales[idx]
                    params = dna[idx]
                    quant_params = round_tensor((params * bn_scale.cpu()) / quant_scale)
                    if quant_params > 127:
                        overflow_num += 1
                        quant_params = 127
                    org_qp = self.org_quantParam[idx]
                    dist = self.hamming_distance_int8(quant_params, org_qp)

                    dist_list.append(dist)
                    qp_list.append((org_qp, quant_params))
                    total_hamming_distance += dist

                total_hamming_distance_list.append(total_hamming_distance)
                total_overflow_number_list.append(overflow_num)
                tmp.append(dist_list)
                total_org_quant_qp_list.append(qp_list)

            self.kernel_swap(dna)
            # Measure the maximum logit/probability perturbation.
            val = self.get_lastFCValue() - self.initialLastFCValue
            if self.attack_catego == -1:
                val_list.append(max(val))
            else:
                val_list.append(val[self.attack_catego])

        return np.array(val_list), total_hamming_distance_list, total_overflow_number_list, tmp, total_org_quant_qp_list

    def kill_bad_according_FCVal(self, kids):
        for key in ['DNA', 'mut_strength']:
            self.pop[key] = np.vstack((self.pop[key], kids[key]))
        fcVal_list, total_hamming_distance_list, total_overflow_number_list, dist, total_org_quant_qp_list = self.get_catego_val()
        fcVal_list_cpy = fcVal_list.copy()                  # Keep the raw FC values for best_predict.
        fitness = self.get_fitness_catego(fcVal_list_cpy, total_hamming_distance_list, total_overflow_number_list, dist)

        idx = np.arange(self.pop['DNA'].shape[0])
        good_idx = idx[fitness.argsort()][:self.popSize]

        for key in ['DNA', 'mut_strength']:
            self.pop[key] = self.pop[key][good_idx]

        # print("best Deviation = " + str(fitness[good_idx[0]])[0:7], str(fitness[good_idx[1]])[0:7],
        #       str(fitness[good_idx[2]])[0:7])  # Keep a compact numeric display.
        self.best_fitness.append(fitness[good_idx[0]])
        self.best_predict.append(fcVal_list[good_idx[0]])
        if total_overflow_number_list != []:
            spacing = 11
            best_dist = ' '.join(f'{x:>{spacing}}' for x in dist[good_idx[0]])
            print("fcVal: " + str(self.best_predict[-1]) + ", hamming dist: " + str(total_hamming_distance_list[good_idx[0]]) + ", Overflow: " + str(total_overflow_number_list[good_idx[0]]))
            print(total_org_quant_qp_list[good_idx[0]])
            print(best_dist)
        else:
            print("fcVal: " + str(self.best_predict[-1]))

    def make_kid(self):
        kids = {'DNA': np.empty((int(self.numKid), self.dnaSize))}
        kids['mut_strength'] = np.empty_like(kids['DNA'])
        for kv, ks in zip(kids['DNA'], kids['mut_strength']):
            # crossover (roughly half p1 and half p2)
            p1, p2 = np.random.choice(np.arange(self.popSize), size=2, replace=False)  # Parents must be different individuals.
            cp = np.random.randint(0, 2, self.dnaSize, dtype=bool)  # crossover points
            kv[cp] = self.pop['DNA'][p1, cp]
            kv[~cp] = self.pop['DNA'][p2, ~cp]
            ks[cp] = self.pop['mut_strength'][p1, cp]
            ks[~cp] = self.pop['mut_strength'][p2, ~cp]

            # mutate (change DNA based on normal distribution)
            ks[:] = np.maximum(ks + (np.random.randn(*ks.shape)), 0.)
            kv += (ks * np.random.randn(*kv.shape)) * 0.05  # Whether 0.05 is the best ?
            # kv[:] = np.clip(kv, *self.popBound)

            # Clamp each generated DNA segment to its layer-specific bounds.
            start = 0
            last_layer = 0
            next_popBound_idx = 0
            for i in range(len(self.attack_layer_dna_size)):
                for j in range(self.attack_layer_dna_size[i][1]):
                    if i == 0:
                        last_layer = self.attack_layer_dna_size[i][4]

                    end = start + self.attack_layer_dna_size[i][5]

                    if self.attack_layer_dna_size[i][4] == last_layer:
                        min_bound = self.popBound[next_popBound_idx][0]
                        max_bound = self.popBound[next_popBound_idx][1]
                        kv[start:end] = np.clip(kv[start:end], min_bound, max_bound)
                    else:
                        last_layer = self.attack_layer_dna_size[i][4]
                        next_popBound_idx += 1
                        min_bound = self.popBound[next_popBound_idx][0]
                        max_bound = self.popBound[next_popBound_idx][1]
                        kv[start:end] = np.clip(kv[start:end], min_bound, max_bound)

                    start = start + self.attack_layer_dna_size[i][5]
            pass

        return kids
