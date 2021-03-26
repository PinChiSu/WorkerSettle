# In[0]:

'''
Optimization module
呼叫範例:
optModel(6, [1,2,3], 'C:/Users/cherc/Desktop/necsys/movetime.xlsx', 'C:/Users/cherc/Desktop/necsys/expectedCalls.xlsx',\
 'C:/Users/cherc/Desktop/necsys/historyCalls.xlsx', 'C:/Users/cherc/Desktop/necsys/SiteInfo.xlsx')
'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from gurobipy import *
from datetime import datetime, date, timedelta
import time
import copy

# regression: history calls and employees
def reg(df_historyCalls):
    X_reg = df_historyCalls['總服務次數'].values.reshape(-1, 1)
    y_reg = df_historyCalls['員工數'].values
    reg = LinearRegression().fit(X_reg, y_reg)
    g = reg.coef_[0]
    s = reg.intercept_

    return g, s

def updateSLAtable(df_reachable, df_needAdjustOK, df_officeMapping): 
    for idx in df_needAdjustOK.index:

        cusID = df_needAdjustOK['CustomerID'].iloc[idx]
        loc = df_needAdjustOK['location'].iloc[idx]
        mapping_idx = df_officeMapping.index[df_officeMapping['name']==loc].tolist()[0]
        cus_Site = df_officeMapping['name'].iloc[mapping_idx]
        cus_idx = df_reachable.index[df_reachable['客戶ID']==cusID].tolist()[0]           
        df_reachable[cus_Site].iloc[cus_idx] = True
        
    return df_reachable

# --------------------------------------------------------------------------------------
# Heuristic Algorithm會用到
def redistribute(h_y,h_w,h,A,d,numofCustomer,numofSite,g,s):
    limit_w = np.zeros([numofSite])
    full_w = np.zeros([numofSite])
    avoid_same_list = np.zeros([numofSite])
    redis_need = True
    Report_back = 0
    all_service_fee = 0
    temp_site = 0

    for j in range(numofSite):
        if np.sum(h_y[:,j]) >= 1:
            service_w = np.where(h_y[:,j] == 1)
            service_w = service_w[0]         
            for k in service_w:               
                limit_w[j] += h[k]
        if limit_w[j] * g + s >= h_w[j]:
            full_w[j] = 1
        else:
            full_w[j] = 0
    
    for i in range(numofCustomer):
        if np.sum(h_y[i,:]) == 1:           
            temp_site = np.where(h_y[i,:] == 1)
            if full_w[temp_site] == 0:
                redis_need = False     
            
        if redis_need:
            for z in range(numofSite):
                avoid_same_list[z] = d[i,z] + 0.01 * z           
            avoid_same_list = sorted(avoid_same_list,reverse = False)
            for k in avoid_same_list:
                for z in range(numofSite):
                    if d[i,z] + 0.01 * z == k:
                        new_y = z
                        break               
                if full_w[new_y] == 0 and A[i,new_y] == 1:
                    if (limit_w[new_y] + h[i]) * g + s <= h_w[new_y]:
                        if np.sum(h_y[i,:]) >= 1:
                            for j in range(numofSite):
                                if h_y[i,j] == 1:
                                    h_y[i,j] = 0                      
                                    limit_w[j] -= h[i]                       
                        h_y[i,new_y] = 1                                               
                        limit_w[new_y] += h[i]                                   
                        break              
            if not np.sum(h_y[i,:]) == 1:
                for k in avoid_same_list:
                    for z in range(numofSite):
                        if d[i,z] + 0.01 * z == k and A[i,z] == 1:
                            new_y = z
                            break
                    for z in range(numofSite):
                        if h_y[i,z] == 1:                                            
                            h_y[i,z] = 0                      
                            limit_w[z] -= h[i]                       
                    h_y[i,new_y] = 1                                      
                    limit_w[new_y] += h[i]                                              
                    break

        for j in range(numofSite):
            if h_y[i,j] == 1:
                all_service_fee += h[i] * d[i,j] * oilprice

        redis_need = True

    limit_w = np.zeros([numofSite])
    for j in range(numofSite):
        if np.sum(h_y[:,j]) >= 1:
            service_w = np.where(h_y[:,j] == 1)
            service_w = service_w[0]         
            for k in service_w:               
                limit_w[j] += h[k]
        if limit_w[j] * g + s >= h_w[j]:
            full_w[j] = 1
        else:
            full_w[j] = 0

    # for j in range(numofSite):
    #     if limit_w[j] * g + s >= h_w[j]:
    #         full_w[j] = 1
    #     else:
    #         full_w[j] = 0

    if np.sum(full_w) >= 1:
        Report_back = 1

    return h_y,Report_back,all_service_fee

# --------------------------------------------------------------------------------------
def LR_subG(customers, locations, scales, oilprice, M, c, f, h, d, A, g, s, isReserved, Lambda):

    # Start
    time_start = time.time()


    # model
    model = Model('Integer Program')

    # disable outputflag
    model.setParam('OutputFlag', 0)

    # variable
    x = model.addVars(scales, vtype=GRB.BINARY, name='x')
    y = model.addVars(customers, vtype=GRB.BINARY, name='y')
    w = model.addVar(vtype=GRB.INTEGER, name='w')

    # constraints
    model.addConstrs(y[i] <= quicksum(x[k] for k in scales) for i in customers)
    model.addConstr(quicksum(x[k] for k in scales) <= 1)
    model.addConstr(w <= x[0] + M * x[1])
    model.addConstr(w >= x[0] + x[1])
    model.addConstr(w >= g * (quicksum(h[i] * y[i] for i in customers)) + s)
    model.addConstr(x[1] >= isReserved) 
                
    # update model
    model.update()

    # objective function: minimize total cost
    obj = quicksum((h[i] * d[i] - Lambda[i] * A[i]) * oilprice * y[i] for i in customers) + f[1] * x[1] + c * w + quicksum(Lambda[i] for i in customers)
    model.setObjective(obj, GRB.MINIMIZE)

    # solve
    model.optimize()

    # result
    sol_x = []
    sol_y = []
    sol_w = 0
    objValue = model.objVal

    for v in model.getVars():
        if v.varName[0] == 'x':
            sol_x.append(v.x)
        elif v.varName[0] == 'y':
            sol_y.append(v.x)
        elif v.varName[0] == 'w':
            sol_w = v.x
    
    time_end = time.time()

    timeG = time_end - time_start

    # 回傳 type：
    # sol_x = [x0, x1], binary, 代表蓋了哪種 scale
    # sol_y = [num of customers], binary
    # sol_w: int, 代表派給這個設施的人數
    # objValue: 這個小 gurobi 算出來的 objective value，其實可忽略
    return sol_x, sol_y, sol_w, objValue, timeG


# Heuristic
def LR_subH(customers, locations, scales, oilprice, M, c, f, h, d, A, g, s, isReserved, Lambda):

    # 這其實是一個背包問題
    # 先把 CP 值排序

    time_start = time.time()


    CP = {}
    for i in range(len(customers)):
        value = (h[i] * d[i] - Lambda[i] * A[i]) * oilprice
        if h[i] != 0:
            CP[i] = value / h[i]
    CP = {k:v for k, v in sorted(CP.items(), key=lambda item: item[1])}
    # for k in CP:
    #     print(k, CP[k])
    
    # #debug
    # result_H = open("testH.csv", "w", newline = "")
    # wH = csv.writer(result_H)
    # wH.writerow(["k", "CP"])
    # for k in CP:
    #     wH.writerow([k, CP[k]])

    # #debug_end

    # 開始一個一個塞，存需要幾個人的時候值 obj 值是多少
    objW = {0:{"value": 0, "weight":0, "chosen":[]}, 1:{"value": c, "weight":0, "chosen":[]}}

    w = 1
    for k in CP:
        # 如果放不下了，服務人員 (w) 就 +1     
        if (objW[w]["weight"] + h[k]) > ((w - s) / g):
            if w + 1 > M:
                break
            else:
                w += 1
                objW[w] = {"value": objW[w - 1]["value"] + c, "weight": objW[w - 1]["weight"], "chosen":copy.deepcopy(objW[w - 1]["chosen"])}
                                
        objW[w]["weight"] += h[k]
        objW[w]["value"] += CP[k] * h[k]
        objW[w]["chosen"].append(k)


    # 找最小
    minValue = 0

    sol_x = [0, 0]
    sol_y = [0 for i in range(len(customers))]
    sol_w = 0


    if isReserved:
        # 保留的話 x 必為 [0,1]
        sol_x = [0, 1]
        minValue = objW[1]["value"]
        sol_w = 1

        # 找 w
        for w in objW:
            if w > 0 :
                if objW[w]["value"] < minValue:
                    sol_w = w
                    minValue = objW[w]["value"]

        # 修正成本、存 minValue
        minValue = objW[sol_w]["value"] + f[1]

    else:
        # 先找 w
        for w in objW:
            if objW[w]["value"] < minValue:
                sol_w = w
                minValue = objW[w]["value"]
        print("M:",M)

        # 存 sol_x
        if sol_w > 0:
            if sol_w == 1:
                sol_x = [1, 0]
            elif sol_w > 1:
                sol_x = [0, 1]

        # 存 minValue
        minValue = objW[sol_w]["value"]
        
        # 存 obj_y
    for i in objW[sol_w]["chosen"]:
        sol_y[i] = 1

    minValue += sum(Lambda)

    time_end = time.time()

    timeH = time_end - time_start
    

    return sol_x, sol_y, sol_w, minValue, timeH

# def LR_subH(customers, locations, scales, oilprice, M, c, f, h, d, A, g, s, isReserved, Lambda):

#     # 這其實是一個背包問題
#     # 先把 CP 值排序

#     time_start = time.time()


#     CP = {}
#     for i in range(len(customers)):
#         value = (h[i] * d[i] - Lambda[i] * A[i]) * oilprice
#         if value < 0:
#             CP[i] = value / h[i]
#     CP = {k:v for k, v in sorted(CP.items(), key=lambda item: item[1])}
    
#     # #debug
#     # result_H = open("testH.csv", "w", newline = "")
#     # wH = csv.writer(result_H)
#     # wH.writerow(["k", "CP"])
#     # for k in CP:
#     #     wH.writerow([k, CP[k]])

#     # #debug_end

#     # 開始一個一個塞，存需要幾個人的時候值 obj 值是多少
#     objW = {0:{"value": 0, "weight":0, "chosen":[]}, 1:{"value": c, "weight":0, "chosen":[]}}

#     w = 1
#     for k in CP:
#         # 如果放不下了，服務人員 (w) 就 +1     
#         if (objW[w]["weight"] + h[k]) > ((w - s) / g):
#             if w + 1 > M:
#                 break
#             else:
#                 w += 1
#                 objW[w] = {"value": objW[w - 1]["value"] + c, "weight": objW[w - 1]["weight"], "chosen":copy.deepcopy(objW[w - 1]["chosen"])}
#                 if w == 2:
#                     objW[w]["value"] += f[1]
                        
        
#         objW[w]["weight"] += h[k]
#         objW[w]["value"] += CP[k] * h[k]
#         objW[w]["chosen"].append(k)


#     # 找最小
#     minValue = 0

#     sol_x = [0, 0]
#     sol_y = [0 for i in range(len(customers))]
#     sol_w = 0

#     ignore = []
#     if isReserved:
#         ignore = [0, 1]
#         minValue = objW[2]["value"]
#         sol_w = 2


#     for w in objW:
#         if w not in ignore:
#             if objW[w]["value"] < minValue:
#                 sol_w = w
#                 minValue = objW[w]["value"]
#     #         print("w:", w, "value:", objW[w]["value"], "weight:", objW[w]["weight"], "numI:", len(objW[w]["chosen"]), "capacity:", (w - s)/g)
#     # print("M:",M)

#     # 存 sol_x
#     if sol_w > 0:
#         if sol_w == 1:
#             sol_x = [1, 0]
#         elif sol_w > 1:
#             sol_x = [0, 1]

#         # 存 minValue
#         minValue = objW[sol_w]["value"]
        
#         # 存 obj_y
#         for i in objW[sol_w]["chosen"]:
#             sol_y[i] = 1

#     minValue += sum(Lambda)

#     time_end = time.time()

#     timeH = time_end - time_start
    

#     return sol_x, sol_y, sol_w, minValue, timeH

# --------------------------------------------------------------------------------------




def optModel(oilprice, reservationSite, reachablePath, needAdjustOKPath, movetimePath, expectedCallsPath, historyCallsPath, siteInfoPath, officeMappingPath, method):
    '''
    <input>
    oilprice: float
    reservationSite: list
    reachablePath: 客戶服務水準滿足表
    needAdjustOKPath: 
    movetimePath, expectedCallsPath, historyCallsPath, siteInfoPath: path
    method: 'M' , 'H' or 'B' . M stands for model, H stands for heuristic, B stands for both.
    <output>
    df_site: table
    dict_assign: dictionary
    '''

    # read data files
    df_movetime = pd.read_excel(movetimePath)
    df_expectedCalls = pd.read_excel(expectedCallsPath)
    df_siteInfo = pd.read_excel(siteInfoPath)
    df_reachable = pd.read_excel(reachablePath)
    df_historyCalls = pd.read_excel(historyCallsPath) 
    df_officeMapping = pd.read_excel(officeMappingPath)
    df_needAdjustOK = pd.read_excel(needAdjustOKPath)

    df_movetime.astype({'客戶ID': 'str'}).dtypes

    siteName = df_siteInfo['據點'].values
    customerID = df_movetime['客戶ID'].values
    df_reachableOK = updateSLAtable(df_reachable, df_needAdjustOK, df_officeMapping)

    # Parameter processing
    M = df_siteInfo['最大容納人數'].values
    c = df_siteInfo['每人年成本'].values
    f = df_siteInfo[['前進據點成本','固定據點成本']].values
    h = df_expectedCalls['預期年服務次數'].values
    d = df_movetime[df_movetime.columns[3:]].values
    A = df_reachableOK[df_reachableOK.columns[3:]].values
    
    numofSite = len(df_siteInfo)
    numofCustomer = len(df_expectedCalls)
    g, s = reg(df_historyCalls)

    scales=[0,1]
    locations=[i for i in range(numofSite)]
    customers=[i for i in range(numofCustomer)]

    reservationSite_idx = []
    for rsvSite in reservationSite:
        reservationSite_idx.append(df_officeMapping.index[df_officeMapping['name']==rsvSite].tolist()[0])

    # --------------------------------------------------------------------------------------
    # OR Algorithm

    if method == 'M':
        
        print(time.asctime(time.localtime(time.time())))
        print("OR模型開始")

        # model
        model = Model('Integer Program')

        # variable
        x = model.addVars(locations, scales, vtype=GRB.BINARY, name='x')
        y = model.addVars(customers, locations, vtype=GRB.BINARY, name='y')
        w = model.addVars(locations, vtype=GRB.INTEGER, name='w')

        # constraints
        model.addConstrs(y[i,j] <= quicksum(x[j,k] for k in scales) for i in customers for j in locations)
        model.addConstrs(quicksum(x[j,k] for k in scales) <= 1 for j in locations)
        model.addConstrs(quicksum(A[i,j]*y[i,j] for j in locations) == 1 for i in customers)
        model.addConstrs(w[j] <= x[j,0]+M[j]*x[j,1] for j in locations)
        model.addConstrs(w[j] >= x[j,0]+x[j,1] for j in locations)
        model.addConstrs(w[j] >= g*(quicksum(h[i]*y[i,j] for i in customers))+s for j in locations)
        model.addConstrs(x[j,1] == 1 for j in reservationSite_idx) 
                    
        # update model
        model.update()

        # objective function: minimize total cost
        obj = quicksum(h[i]*oilprice*d[i,j]*y[i,j] for i in customers for j in locations)+quicksum(f[j,1]*x[j,1] for j in locations)+quicksum(c[j]*w[j] for j in locations)
        model.setObjective(obj, GRB.MINIMIZE)

        # solve
        model.optimize()

        # outcome display
        siteScale = ['不蓋據點' for i in range(numofSite)]
        assignSite = []
        siteEmp = []

        count=0 
        for v in model.getVars():
            var_result = v.Varname
            if v.x == 1:
                if var_result[0] == 'x':
                    if int(var_result.split(',')[-1].strip(']')) == 0:
                        siteScale[int(count/2)] = '前進據點'
                    elif int(var_result.split(',')[-1].strip(']')) == 1:
                        siteScale[int(count/2)] = '固定據點'
                if var_result[0] == 'y':
                    assignSite.append(int((var_result.split(','))[-1].strip(']')))
            if var_result[0] == 'w':
                siteEmp.append(int(v.x))
            if count < numofSite*2:
                count = count+1

        df_site = pd.DataFrame(siteName, columns = ['據點'])

        df_site['規模'] = siteScale
        df_site['員工數'] = siteEmp

        assignSiteName = []
        for a in assignSite:
            assignSiteName.append(siteName[a])
        df_assign = pd.DataFrame(customerID, columns=['客戶ID'])

        df_assign['指派據點'] = assignSiteName
        dict_assign = {}
        for site in siteName:
            dict_assign[site]=df_assign[df_assign['指派據點']==site]

        setCost = [0 for i in range(numofSite)]
        empCost = [0 for i in range(numofSite)]
        serviceCost = [0 for i in range(numofSite)]
        totalCost = [0 for i in range(numofSite)]
        annualCalls = [0 for i in range(numofSite)]

        for i in range(len(siteName)):
            site = siteName[i]
            if df_site['規模'][df_site.index[df_site['據點']==site].tolist()[0]] != '不蓋據點':
                sCost = 0
                for j in range(len(dict_assign[site]['客戶ID'])):
                    cusid = dict_assign[site]['客戶ID'].iloc[j]
                    cusidx = list(df_movetime.index[df_movetime['客戶ID']==cusid])[0] 
                    mt = df_movetime[site].iloc[cusidx]
                    sCost += mt*oilprice*df_expectedCalls['預期年服務次數'].iloc[cusidx]

                serviceCost[i] = int(round(sCost))
                
        # 增加各據點每年總服務次數 = 加總客戶預期服務次數
        for i in range(len(siteName)):
            sumofCalls = 0
            for cus in dict_assign[siteName[i]]['客戶ID']:
                sumofCalls += float(df_expectedCalls['預期年服務次數'][df_expectedCalls.index[df_expectedCalls['客戶ID']==cus].tolist()[0]])
            annualCalls[i] = round(sumofCalls)

        df_site['建置成本($)']=setCost
        df_site['服務成本($)']=serviceCost
        df_site['員工成本($)']=empCost
        df_site['總成本($)']=totalCost
        df_site['年度總服務次數']=annualCalls

        for idx in df_site.index:
            if df_site['規模'].iloc[idx] == '固定據點':
                df_site['建置成本($)'].iloc[idx] = int(round(df_siteInfo['固定據點成本'].iloc[idx]))
            elif df_site['規模'].iloc[idx] == '前進據點':
                df_site['建置成本($)'].iloc[idx] = int(round(df_siteInfo['前進據點成本'].iloc[idx]))
            else:
                df_site['建置成本($)'].iloc[idx] = 0
            
            df_site['員工成本($)'].iloc[idx] = int(round(df_site['員工數'].iloc[idx]*df_siteInfo['每人年成本'].iloc[idx]))
                
            df_site['總成本($)'].iloc[idx] = int(round(df_site['建置成本($)'].iloc[idx]+df_site['員工成本($)'].iloc[idx]+df_site['服務成本($)'].iloc[idx]))

        for idx in df_site.index:
            df_site['建置成本($)'].iloc[idx] = format(df_site['建置成本($)'].iloc[idx], ',')
            df_site['員工成本($)'].iloc[idx] = format(df_site['員工成本($)'].iloc[idx], ',')
            df_site['服務成本($)'].iloc[idx] = format(df_site['服務成本($)'].iloc[idx], ',')
            df_site['總成本($)'].iloc[idx] = format(df_site['總成本($)'].iloc[idx], ',')
            df_site['年度總服務次數'].iloc[idx] = format(df_site['年度總服務次數'].iloc[idx], ',')

        df_site.to_excel('D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/Objective_Site.xlsx', encoding='utf-8', index=False)
        df_assign.to_excel('D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/Objective_Assign.xlsx', encoding='utf-8', index=False)

        print(time.asctime(time.localtime(time.time())))
        print("OR模型結束")

    # --------------------------------------------------------------------------------------
    # Hueristic Algorithm

    if method == 'H':

        print(time.asctime(time.localtime(time.time())))
        print("Heuristic 演算法開始")

        reserved_objective = [1,1,1,1,1,1,1,1,1,1,1,1]
        for i in range(len(reserved_objective)):
            if reserved_objective[i] >= 2 and not i in reservationSite_idx:
                reservationSite_idx.append(i)

        h_x = np.ones([numofSite])
        h_y = np.zeros([numofCustomer,numofSite])
        h_w = np.zeros([numofSite])
        best_hy = np.zeros([numofCustomer,numofSite])
        obj_set = 0
        max_diff_obj = obj_set
        last_obj = -1
        best_y = -1

        # 決定一個分配程式要跑幾次才會認定原程式無解
        time_index1 = 10
        # 決定一個分配程式要跑幾次才會認定原程式無解
        time_index2 = 5
        # 決定一個裁切員工程式一次要裁切幾(+1)分之一的員工
        time_index3 = 9

        for i in range(numofSite):
            h_w[i] = math.floor(M[i])
            if h_w[i] < reserved_objective[i]:
                h_w[i] = reserved_objective[i]
            # if h_w[i] >= 200:
            #     h_w[i] = 200

        # 第一部份計算修正過後的A值(理論上未來就不用再用它了)
        max_converge_try = 0.8
        try_range_k = 20
        parameter_range_k = max_converge_try-try_range_k/100

        temp = redistribute(h_y,h_w,h,A,d,numofCustomer,numofSite,g,s)

        for t in range(time_index1):
            if temp[1] == 1:
                temp = redistribute(temp[0],h_w,h,A,d,numofCustomer,numofSite,g,s)
            else:
                break
        h_y = temp[0]
        last_obj = temp[2]
        for k in range(numofSite):
            last_obj += f[k,0] * (1-h_x[k]) + f[k,1] * h_x[k] + h_w[k] * c[k]
        print("Distribution Initiate Success")

        temp_A = A
        for k in range(try_range_k):
            for i in range(numofCustomer):
                avg_dis = 0
                for j in range(numofSite):
                    avg_dis += d[i][j]               
                avg_dis = avg_dis / numofSite - (avg_dis / numofSite - np.min(d[i][:])) * (parameter_range_k+k*0.01)
                for j in range(numofSite):
                    if d[i][j] <= avg_dis:
                        A[i][j] = 1
                    else:
                        A[i][j] = 0
                if np.sum(d[i][:] <= 0):
                    print("無法再縮小gap!")
                    break
            temp = redistribute(h_y,h_w,h,A,d,numofCustomer,numofSite,g,s)        
            for t in range(time_index1):
                if temp[1] == 1:
                    temp = redistribute(temp[0],h_w,h,A,d,numofCustomer,numofSite,g,s)
                else:
                    break
            if temp[1] == 0:
                h_y = temp[0]
            else:
                break
        A = temp_A

        for j in range(numofSite):
            max_diff_obj = obj_set
            best_y = -1
            for i in range(numofSite):
                if h_x[i] != 0 and h_w[i] != 1 and (i not in reservationSite_idx):
                    h_x[i] = 0
                    h_w[i] = 1
                    temp = redistribute(h_y,h_w,h,A,d,numofCustomer,numofSite,g,s)
                    for t in range(time_index1):
                        if temp[1] == 1:
                            temp = redistribute(temp[0],h_w,h,A,d,numofCustomer,numofSite,g,s)
                        else:
                            break
                    if temp[1] == 0:
                        this_obj = temp[2]
                        for k in range(numofSite):
                            this_obj += f[k,0] * (1-h_x[k]) + f[k,1] * h_x[k] + h_w[k] * c[k]
                        if last_obj - this_obj > max_diff_obj:
                            max_diff_obj = last_obj - this_obj
                            best_y = i
                            best_hy = temp[0]
                    h_x[i] = 1
                    h_w[i] = math.floor(M[i])

            if best_y != -1:               
                h_x[best_y] = 0
                h_w[best_y] = 1
                h_y = best_hy
                last_obj = last_obj - max_diff_obj
                print("Delete:",best_y+1)
            else:
                break
        print("Delete Fixed Cost Success")

        # temp_A = A
        # for k in range(try_range_k):
        #     for i in range(numofCustomer):
        #         avg_dis = 0
        #         for j in range(numofSite):
        #             avg_dis += d[i][j]               
        #         avg_dis = avg_dis / numofSite - (avg_dis / numofSite - np.min(d[i][:])) * (parameter_range_k+k*0.01)
        #         for j in range(numofSite):
        #             if d[i][j] <= avg_dis:
        #                 A[i][j] = 1
        #             else:
        #                 A[i][j] = 0
        #         if np.sum(d[i][:] <= 0):
        #             print("無法再縮小gap!")
        #             break
        #     temp = redistribute(h_y,h_w,h,A,d,numofCustomer,numofSite,g,s)        
        #     for t in range(time_index1):
        #         if temp[1] == 1:
        #             temp = redistribute(temp[0],h_w,h,A,d,numofCustomer,numofSite,g,s)
        #         else:
        #             break
        #     if temp[1] == 0:
        #         h_y = temp[0]
        #     else:
        #         break
        # A = temp_A

        resize_set = 0
        max_diff_resize = resize_set
        last_resize = last_obj
        best_resize = -1
        this_resize = -1
        for j in range(numofSite * time_index3):
            max_diff_resize = resize_set
            best_resize = -1
            for i in range(numofSite):
                if h_w[i] > 1 and math.floor(h_w[i] * (1 - 1 / (time_index3 + 1))) >= reserved_objective[i]:                   
                    temp_hw = h_w[i]
                    #h_w[i] = h_w[i] - math.floor(M[i] / (time_index3 + 1))
                    h_w[i] = math.floor(h_w[i] * (1 - 1 / (time_index3 + 1)))
                    if h_w[i] <= 0:
                        h_w[i] = 1
                    temp = redistribute(h_y,h_w,h,A,d,numofCustomer,numofSite,g,s)        
                    for t in range(time_index2):
                        if temp[1] == 1:
                            temp = redistribute(temp[0],h_w,h,A,d,numofCustomer,numofSite,g,s)
                        else:
                            break
                    if temp[1] == 0:
                        this_resize = temp[2]
                        for k in range(numofSite):
                            this_resize += f[k,0] * (1-h_x[k]) + f[k,1] * h_x[k] + h_w[k] * c[k]
                        if last_resize - this_resize > max_diff_resize:
                            max_diff_resize = last_resize - this_resize
                            best_resize = i
                            best_hy = temp[0]
                            traffic_cost = temp[2]                                                                
                    h_w[i] = temp_hw

            if best_resize != -1:
                #h_w[best_resize] = h_w[best_resize] - math.floor(M[best_resize] / (time_index3 + 1))
                h_w[best_resize] = math.floor(h_w[best_resize] * (1 - 1 / (time_index3 + 1)))
                h_y = best_hy
                last_resize = last_resize - max_diff_resize
                print("Resize:",best_resize+1,"to",h_w[best_resize])
            else:
                break
        print("Resize Success")

        # 最後再檢查一次gap
        # 計算修正過後的A值(理論上未來就不用再用它了)

        for k in range(try_range_k):
            for i in range(numofCustomer):
                avg_dis = 0
                for j in range(numofSite):
                    avg_dis += d[i][j]               
                avg_dis = avg_dis / numofSite - (avg_dis / numofSite - np.min(d[i][:])) * (parameter_range_k+k*0.01)
                for j in range(numofSite):
                    if d[i][j] <= avg_dis:
                        A[i][j] = 1
                    else:
                        A[i][j] = 0
                if np.sum(d[i][:] <= 0):
                    print("無法再縮小gap!")
                    break
            temp = redistribute(h_y,h_w,h,A,d,numofCustomer,numofSite,g,s)        
            for t in range(time_index1):
                if temp[1] == 1:
                    temp = redistribute(temp[0],h_w,h,A,d,numofCustomer,numofSite,g,s)
                else:
                    break
            if temp[1] == 0:
                h_y = temp[0]
            else:
                break
        
        for j in range(numofSite):
            temp_check = 0
            for i in range(numofCustomer):           
                if h_y[i][j] == 1:
                    temp_check += h[i]
            h_w[j] = math.ceil(temp_check * g + s)
            if h_w[j] <= reserved_objective[j]:
                h_w[j] = reserved_objective[j]

        # --------------------------------------------------------------------------------------
        # 以下部分則接回原本的程式

        # outcome display
        siteScale = []
        assignSite = []
        siteEmp = []    

        for x in h_x:
            if x == 0:
                siteScale.append('前進據點')
            if x == 1:
                siteScale.append('固定據點')

        for i in range(numofCustomer):
            temp3 = np.where(h_y[i,:] == 1)
            temp3 = temp3[0]
            temp3 = temp3[0]
            assignSite.append(temp3)

        # print(assignSite)

        for w in h_w:
            siteEmp.append(w)      

        df_site = pd.DataFrame(siteName, columns = ['據點'])

        df_site['規模'] = siteScale
        df_site['員工數'] = siteEmp

        assignSiteName = []
        for a in assignSite:
            assignSiteName.append(siteName[a])
        df_assign = pd.DataFrame(customerID, columns=['客戶ID'])
        
        # print('assignSiteName:',len(assignSiteName))
        # print('assignSite:',len(assignSite))
        # print('customerID:',len(customerID))

        df_assign['指派據點'] = assignSiteName
        dict_assign = {}
        for site in siteName:
            dict_assign[site]=df_assign[df_assign['指派據點']==site]

        setCost = [0 for i in range(numofSite)]
        empCost = [0 for i in range(numofSite)]
        serviceCost = [0 for i in range(numofSite)]
        totalCost = [0 for i in range(numofSite)]
        annualCalls = [0 for i in range(numofSite)]

        for i in range(len(siteName)):
            site = siteName[i]
            if df_site['規模'][df_site.index[df_site['據點']==site].tolist()[0]] != '不蓋據點':
                sCost = 0
                for j in range(len(dict_assign[site]['客戶ID'])):
                    cusid = dict_assign[site]['客戶ID'].iloc[j]
                    cusidx = list(df_movetime.index[df_movetime['客戶ID']==cusid])[0] 
                    mt = df_movetime[site].iloc[cusidx]
                    sCost += mt*oilprice*df_expectedCalls['預期年服務次數'].iloc[cusidx]

                serviceCost[i] = int(round(sCost))
                
        # 增加各據點每年總服務次數 = 加總客戶預期服務次數
        for i in range(len(siteName)):
            sumofCalls = 0
            for cus in dict_assign[siteName[i]]['客戶ID']:
                sumofCalls += float(df_expectedCalls['預期年服務次數'][df_expectedCalls.index[df_expectedCalls['客戶ID']==cus].tolist()[0]])
            annualCalls[i] = round(sumofCalls)

        df_site['建置成本($)']=setCost
        df_site['服務成本($)']=serviceCost
        df_site['員工成本($)']=empCost
        df_site['總成本($)']=totalCost
        df_site['年度總服務次數']=annualCalls

        for idx in df_site.index:
            if df_site['規模'].iloc[idx] == '固定據點':
                df_site['建置成本($)'].iloc[idx] = int(round(df_siteInfo['固定據點成本'].iloc[idx]))
            elif df_site['規模'].iloc[idx] == '前進據點':
                df_site['建置成本($)'].iloc[idx] = int(round(df_siteInfo['前進據點成本'].iloc[idx]))
            else:
                df_site['建置成本($)'].iloc[idx] = 0
            
            df_site['員工成本($)'].iloc[idx] = int(round(df_site['員工數'].iloc[idx]*df_siteInfo['每人年成本'].iloc[idx]))
                
            df_site['總成本($)'].iloc[idx] = int(round(df_site['建置成本($)'].iloc[idx]+df_site['員工成本($)'].iloc[idx]+df_site['服務成本($)'].iloc[idx]))

        for idx in df_site.index:
            df_site['建置成本($)'].iloc[idx] = format(df_site['建置成本($)'].iloc[idx], ',')
            df_site['員工成本($)'].iloc[idx] = format(df_site['員工成本($)'].iloc[idx], ',')
            df_site['服務成本($)'].iloc[idx] = format(df_site['服務成本($)'].iloc[idx], ',')
            df_site['總成本($)'].iloc[idx] = format(df_site['總成本($)'].iloc[idx], ',')
            df_site['年度總服務次數'].iloc[idx] = format(df_site['年度總服務次數'].iloc[idx], ',')

        # --------------------------------------------------------------------------------------

        df_site.to_excel('D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/Objective_Site_Heuristic.xlsx', encoding='utf-8', index=False)
        df_assign.to_excel('D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/Objective_Assign_Heuristic.xlsx', encoding='utf-8', index=False)

        print(time.asctime(time.localtime(time.time())))
        print("Heuristic 演算法結束")

    # --------------------------------------------------------------------------------------
    # Lagrange OR Model & Lagrange Heuristic Model

    if method == 'LG' or method == 'LH':

        print(time.asctime(time.localtime(time.time())))
        if method == 'LG':
            print("Lagrange OR 演算法開始")
        if method == 'LH':
            print("Lagrange Heuristic 演算法開始")
       
        # 決定一個分配程式要跑幾次才會認定原程式無解
        time_index1 = 5

        # 基本參數設定(SubG)
        isReserved = 1
        temp_d = [d[i][0] for i in range(len(d))]
        temp_A = [A[i][0] for i in range(len(A))]
        # Lambda = [h[i] * oilprice * temp_d[i] * 1.1  for i in range(numofCustomer)]
        Lambda = [500 for i in range(numofCustomer)]

        # 基本參數設定(Redistribution)
        sol_x = np.zeros([numofSite])
        epsilon = 10
        step = np.zeros([numofCustomer])
        undo_turn = 0

        while True:
         
            # 開始準備去跑Subproblem
            sol_y = np.zeros([numofCustomer,numofSite])           
            sol_w = np.zeros([numofSite])
            for j in range(numofSite):
                if method == 'LG':
                    temp_subG = LR_subG(customers, locations, scales, oilprice, M[j], c[j], f[j], h, temp_d, temp_A, g, s, isReserved, Lambda)
                if method == 'LH':
                    temp_subG = LR_subH(customers, locations, scales, oilprice, M[j], c[j], f[j], h, temp_d, temp_A, g, s, isReserved, Lambda)  
                for i in range(numofCustomer):
                    if temp_subG[1][i] == 1:
                        sol_y[i][j] = 1              
                sol_w[j] = temp_subG[2]
            # 跑完Subproblem，開始更新epsilon
            for i in range(numofCustomer):               
                step[i] = epsilon * (1 - np.sum(A[i,:] * sol_y[i,:]))
                Lambda[i] += step[i] 

            temp_Lam = 0
            for z in range(numofCustomer):
                if Lambda[z] >= 0:
                    temp_Lam += 1

            print("computed_num_worker:",sol_w)          
            print("total_cost:",temp_subG[3])
            print("step[0-4]:",step[0]," ",step[1]," ",step[2]," ",step[3]," ",step[4])
            print("total_step:",np.sum(step))
            print("Lambda[0-4]:",Lambda[0]," ",Lambda[1]," ",Lambda[2]," ",Lambda[3]," ",Lambda[4])               
            print("Lambda_Positive:",temp_Lam)
            print("---------------------------------------------------------")           
            undo_turn += 1 
                
            # 目前仍然採用做到一定次數後自動結束的方式
            if undo_turn >= 30:
                print("超過重新搜尋次數限制!")
                break

        # 這部分是結果後處理的部分，不是Lagrange的重點
        temp = redistribute(sol_y,sol_w,h,A,d,numofCustomer,numofSite,g,s)
        if np.sum(sol_w) >= 120:     
            for t in range(time_index1):
                if temp[1] == 1:
                    temp = redistribute(temp[0],sol_w,h,A,d,numofCustomer,numofSite,g,s)
                else:
                    break
        sol_y = temp[0]    

        for j in range(numofSite):
            temp_check = 0
            for i in range(numofCustomer):           
                if sol_y[i][j] == 1:
                    temp_check += h[i]
            sol_w[j] = math.ceil(temp_check * g + s)
            if sol_w[j] <= 1 and (j not in reservationSite_idx):
                sol_x[j] = 0
            else:
                sol_x[j] = 1
        
        print(sol_w)
        print(sol_x)

        # LR_subG(customers, locations, scales, oilprice, M, c, f, h, d, A, g, s, isReserved, Lambda)
        # 回傳 type：
        # sol_x = [x0, x1], binary, 代表蓋了哪種 scale
        # sol_y = [num of customers], binary
        # sol_w: int, 代表派給這個設施的人數
        # objValue: 這個小 gurobi 算出來的 objective value，其實可忽略

        # --------------------------------------------------------------------------------------
        # 以下部分則接回原本的程式

        # outcome display
        siteScale = []
        assignSite = []
        siteEmp = []    

        for x in sol_x:
            if x == 0:
                siteScale.append('前進據點')
            if x == 1:
                siteScale.append('固定據點')

        for i in range(numofCustomer):
            temp3 = np.where(sol_y[i,:] == 1)
            temp3 = temp3[0]
            temp3 = temp3[0]
            assignSite.append(temp3)

        # print(assignSite)

        for w in sol_w:
            siteEmp.append(w)      

        df_site = pd.DataFrame(siteName, columns = ['據點'])

        df_site['規模'] = siteScale
        df_site['員工數'] = siteEmp

        assignSiteName = []
        for a in assignSite:
            assignSiteName.append(siteName[a])
        df_assign = pd.DataFrame(customerID, columns=['客戶ID'])
        
        # print('assignSiteName:',len(assignSiteName))
        # print('assignSite:',len(assignSite))
        # print('customerID:',len(customerID))

        df_assign['指派據點'] = assignSiteName
        dict_assign = {}
        for site in siteName:
            dict_assign[site]=df_assign[df_assign['指派據點']==site]

        setCost = [0 for i in range(numofSite)]
        empCost = [0 for i in range(numofSite)]
        serviceCost = [0 for i in range(numofSite)]
        totalCost = [0 for i in range(numofSite)]
        annualCalls = [0 for i in range(numofSite)]

        for i in range(len(siteName)):
            site = siteName[i]
            if df_site['規模'][df_site.index[df_site['據點']==site].tolist()[0]] != '不蓋據點':
                sCost = 0
                for j in range(len(dict_assign[site]['客戶ID'])):
                    cusid = dict_assign[site]['客戶ID'].iloc[j]
                    cusidx = list(df_movetime.index[df_movetime['客戶ID']==cusid])[0] 
                    mt = df_movetime[site].iloc[cusidx]
                    sCost += mt*oilprice*df_expectedCalls['預期年服務次數'].iloc[cusidx]

                serviceCost[i] = int(round(sCost))
                
        # 增加各據點每年總服務次數 = 加總客戶預期服務次數
        for i in range(len(siteName)):
            sumofCalls = 0
            for cus in dict_assign[siteName[i]]['客戶ID']:
                sumofCalls += float(df_expectedCalls['預期年服務次數'][df_expectedCalls.index[df_expectedCalls['客戶ID']==cus].tolist()[0]])
            annualCalls[i] = round(sumofCalls)

        df_site['建置成本($)']=setCost
        df_site['服務成本($)']=serviceCost
        df_site['員工成本($)']=empCost
        df_site['總成本($)']=totalCost
        df_site['年度總服務次數']=annualCalls

        for idx in df_site.index:
            if df_site['規模'].iloc[idx] == '固定據點':
                df_site['建置成本($)'].iloc[idx] = int(round(df_siteInfo['固定據點成本'].iloc[idx]))
            elif df_site['規模'].iloc[idx] == '前進據點':
                df_site['建置成本($)'].iloc[idx] = int(round(df_siteInfo['前進據點成本'].iloc[idx]))
            else:
                df_site['建置成本($)'].iloc[idx] = 0
            
            df_site['員工成本($)'].iloc[idx] = int(round(df_site['員工數'].iloc[idx]*df_siteInfo['每人年成本'].iloc[idx]))
                
            df_site['總成本($)'].iloc[idx] = int(round(df_site['建置成本($)'].iloc[idx]+df_site['員工成本($)'].iloc[idx]+df_site['服務成本($)'].iloc[idx]))

        for idx in df_site.index:
            df_site['建置成本($)'].iloc[idx] = format(df_site['建置成本($)'].iloc[idx], ',')
            df_site['員工成本($)'].iloc[idx] = format(df_site['員工成本($)'].iloc[idx], ',')
            df_site['服務成本($)'].iloc[idx] = format(df_site['服務成本($)'].iloc[idx], ',')
            df_site['總成本($)'].iloc[idx] = format(df_site['總成本($)'].iloc[idx], ',')
            df_site['年度總服務次數'].iloc[idx] = format(df_site['年度總服務次數'].iloc[idx], ',')

        # --------------------------------------------------------------------------------------
        if method == 'LG':
            df_site.to_excel('D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/Objective_Site_LagrangeG.xlsx', encoding='utf-8', index=False)
            df_assign.to_excel('D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/Objective_Assign_LagrangeG.xlsx', encoding='utf-8', index=False)
        if method == 'LH':
            df_site.to_excel('D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/Objective_Site_LagrangeH.xlsx', encoding='utf-8', index=False)
            df_assign.to_excel('D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/Objective_Assign_LagrangeH.xlsx', encoding='utf-8', index=False)
        print(time.asctime(time.localtime(time.time())))
        print("Lagrange 演算法結束")

    return 0

# In[1]:

oilprice = 6
reservationSite = ["南港","桃園","台中","高雄"]
# reachablePath = "D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/Reachable.xlsx"
# needAdjustOKPath = "D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/NeedAdjustOK.xlsx"
# movetimePath = "D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/Movetime.xlsx"
# expectedCallsPath = "D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/ExpectedCalls.xlsx"
# historyCallsPath = "D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/HistoryCalls.xlsx"
# siteInfoPath = "D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/SiteInfo.xlsx"
# officeMappingPath = "D:/學校文件-碩士/碩一下/寒假論文/專案2-派工與前進據點/OfficeMapping.xlsx"
reachablePath = "Reachable.xlsx"
needAdjustOKPath = "NeedAdjustOK.xlsx"
movetimePath = "Movetime.xlsx"
expectedCallsPath = "ExpectedCalls.xlsx"
historyCallsPath = "HistoryCalls.xlsx"
siteInfoPath = "SiteInfo.xlsx"
officeMappingPath = "OfficeMapping.xlsx"
method = 'H'
# method = 'M' or 'H' or 'LG' or 'LH'
optModel(oilprice, reservationSite, reachablePath, needAdjustOKPath, movetimePath, expectedCallsPath, historyCallsPath, siteInfoPath, officeMappingPath, method)

# In[2]:
