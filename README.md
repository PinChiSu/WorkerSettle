# 電梯派遣人員優化專案
將已知的可到達途經、移動時間、限定派遣組合等條件作為Input輸入
的利用Heuristic algo、Gurobi OR與加入Lagrangian Relaxation的Scenario等情形找出人員派遣的最佳解

* 將已知的參數輸入至其中：
  reachablePath:    可到達路徑   
  needAdjustOKPath: 限定派遣組合
  movetimePath:     移動時間
  expectedCallsPath:預期被呼叫次數
  historyCallsPath :歷史呼叫次數
  siteInfoPath     :派遣處資訊
  officeMappingPath:辦公室地點
  
* 演算法：
  演算將會分成Heuistic Algorithm與Gurobi OR作解算，其中又可選擇是否將Lagrangian Relaxation(LG)，Flow Chart如下：
  
 ![image](https://user-images.githubusercontent.com/53268937/112655912-d7612b80-8e8b-11eb-8400-50f879184c26.png)

  即求出最佳解。
