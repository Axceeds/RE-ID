import numpy as np
import file_process as fp
from sequence_process import SequenceProcess
import matrix_process as mp
import matplotlib.pyplot as plt
unprocessed_data = []
processed_data = []
for j in range(1,3):
    for i in range(1,11):
        origin_data = fp.xml_process(file_path=r'D:/2.PycharmProjects/红外数据记录/2022.03.09第一次数据采集与实验/'+str(j)+'/'+str(i)+'.xml')
        unprocessed_data.append(origin_data[0]['side0'])

unprocessed_data= np.array(unprocessed_data)

