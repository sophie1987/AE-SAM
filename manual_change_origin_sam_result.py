import numpy as np


def change_4():
    result = np.loadtxt('data/result4/sam_auto_origin.csv', delimiter=',', dtype=np.int32)
    print(np.unique(result))
    result[result<4] = 60
    result[(result>=4)&(result<7)]=20
    result[result==7] = 70
    result[(result>7) & (result<18)]=20
    result[result==18]=90
    result[(result>18) & (result<22)]=20
    result[result==22]=90
    result[result==23]=20
    np.savetxt('data/result4/sam_auto.csv', result, delimiter=',',fmt='%d')


def change_3():
    result = np.loadtxt('data/result3/sam_auto_origin.csv', delimiter=',', dtype=np.int32)
    print(np.unique(result))
    result[result>0] = 20
    result[result==0] = 30
    np.savetxt('data/result3/sam_auto.csv', result, delimiter=',',fmt='%d')

def change_2():
    result = np.loadtxt('data/result2/sam_auto_origin.csv', delimiter=',', dtype=np.int32)
    print(np.unique(result))
    result[result==1] = 30
    result[result!=30] = 20
    np.savetxt('data/result2/sam_auto.csv', result, delimiter=',',fmt='%d')

def change_1():
    result = np.loadtxt('data/result1/sam_auto_origin.csv', delimiter=',', dtype=np.int32)
    print(np.unique(result))
    result[(result==3) | (result==7)] = 90
    result[result!=90] = 20
    np.savetxt('data/result1/sam_auto.csv', result, delimiter=',',fmt='%d')

change_1()