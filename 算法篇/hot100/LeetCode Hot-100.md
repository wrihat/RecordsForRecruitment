## 1 哈希
### 1.1 leetcode 1两数之和
给定一个整数数组 nums 和一个整数目标值target，请你在该数组中找出和为目标值target的那两个整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。

你可以按任意顺序返回答案。

```cpp
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> umap;
        for(int i = 0; i < nums.size(); i++) {
            int temp = target - nums[i];
            if(umap.find(temp) != umap.end()) {
                return {umap[temp], i};
            } else {
                umap.insert({nums[i], i});
            }
        }
        return {};
    }
```



### 1.2 leetcode 49字母异位词分组
给你一个字符串数组，请你将字母异位词组合在一起。可以按任意顺序返回结果列表。  
字母异位词是由重新排列源单词的所有字母得到的一个新单词。

```cpp
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<int>> umap;
        for(int i = 0; i  <strs.size(); i++) {
            string tempStr = strs[i];
            sort(tempStr.begin(), tempStr.end());
            if(umap.find(tempStr) != umap.end()) {
                umap[tempStr].push_back(i);
            } else {
                umap[tempStr] = {i};
            }
        }
        vector<vector<string>> ans;
        for(auto& item: umap) {
            vector<string> temp;
            for(auto& index: item.second) {
                temp.push_back(strs[index]);
            }
            ans.push_back(temp);
        }
        return ans;
    }
```



### 1.3 leetcode 128最长连续序列
给定一个未排序的整数数组 nums 找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。  
请你设计并实现时间复杂度为O(n)的算法解决此问题。

```cpp
int longestConsecutive(vector<int>& nums){
        if(nums.size() <= 1) return nums.size();
        unordered_set<int> num_set;
        for(auto& num: nums) num_set.insert(num);
        int longestStreak = 0;
        for(auto& numItem: nums) {
            if(num_set.find(numItem - 1) == num_set.end()) {// 关键，前一个不存在
                int tempLen = 0;							//说明在x,x+1,x+2,x+3...,x+y序列中
                int cur = numItem;							//x是第一个元素
                while(num_set.find(cur) != num_set.end()) {
                    tempLen++;
                    cur++;
                }
                longestStreak = max(tempLen, longestStreak);
            } 
        }
        return longestStreak;
}
```



## 2 双指针
### 2.1 leetcode 283移动零
![](https://cdn.nlark.com/yuque/0/2024/png/44540266/1730042972936-2b6cb001-8049-49f6-a6ce-f41d4c6404c6.png)

```cpp
void moveZeroes(vector<int>& nums) {
        int left = 0;
        int right = left;
        while(right < nums.size()) {
            if(nums[right] != 0) {
                nums[left++] = nums[right];
            }
            right++;
        }
        while(left < nums.size()) {
            nums[left] = 0;
            left++;
        }
  }
```

### 2.2 leetcode 11 盛水最多的容器
![](https://cdn.nlark.com/yuque/0/2024/png/44540266/1730043573272-8ef4878a-8832-4618-8a3d-9d19565e5e36.png)

![](https://cdn.nlark.com/yuque/0/2024/png/44540266/1730166161438-87873d68-0276-49be-9036-8530a82bad2e.png)

```cpp
int maxArea(vector<int>& height) {
        int left = 0; 
        int right = height.size() - 1;
        int ansArea = 0;
        while(left <= right) {
            int temp = (right - left) * min(height[left], height[right]);
            ansArea = max(ansArea, temp);
            if(height[left] > height[right]) {
                right--;
            } else {
                left++;
            }
        }
        return ansArea;
    }
```



### 2.3 leetcode 15 三数之和
![](https://cdn.nlark.com/yuque/0/2024/png/44540266/1730221037242-9d71b267-1d4a-4377-a93c-502fa6d4c4b3.png)

![](https://cdn.nlark.com/yuque/0/2024/png/44540266/1730221045601-b352cb01-18f5-46e5-ac12-f749eae3e8b4.png)

```cpp
vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());  //但凡涉及到去重，则大多数需要排序
        vector<vector<int>> ans;
        for(int i = 0; i < nums.size(); i++) {
            if(i > 0 && nums[i] == nums[i - 1]) continue;
            unordered_set<int> used;
            int tempTarget = 0 - nums[i];
            for(int j = i + 1; j < nums.size(); j++) {
                if(j > i + 2 && nums[j] == nums[j-1] && nums[j - 1] ==  nums[j -2]) continue; // 对b元素去重
                if(used.find(tempTarget - nums[j]) != used.end()) {
                    ans.push_back({nums[i], tempTarget - nums[j], nums[j]});
                    used.erase(tempTarget - nums[j]);
                } else {
                    used.insert(nums[j]);
                }
            }
        }
        return ans;
    }
```



### 2.4 leetcode 42 接雨水
![](https://cdn.nlark.com/yuque/0/2024/png/44540266/1730274766420-2abdbab6-541b-489d-a7d2-879123d9c58d.png)

```cpp
int trap(vector<int>& height) {
        //  解法1 双指针暴力匹配，超时
        // int sum = 0; 
        // for(int i = 0; i < height.size(); i++) {
        //     if(i == 0 || i == height.size() - 1) continue;
        //     int rHeight = height[i]; //记录右边柱子的最高高度
        //     int lHeight = height[i]; //记录左边主子的最高高度
        //     for(int r = i + 1; r <height.size(); r++) {
        //         if(height[r] > rHeight) rHeight = height[r];
        //     }
        //     for(int l = i - 1; i >= 0; l--) {
        //         if(height[l] > lHeight) lHeight = height[l];
        //     }
        //     int h = min(lHeight, rHeight) - height[i];
        //     if(h > 0) sum += h;
        // }
        // return sum;
        // 解法2 双指针空间换时间
        // int sum = 0;
        // vector<int> maxLeft(height.size());
        // vector<int> maxRight(height.size());
        // maxLeft[0] = height[0];
        // maxRight[height.size() - 1] = height.back();
        // for(int i = 1; i < height.size();i++) {
        //     maxLeft[i] = max(maxLeft[i - 1], height[i]);
        // }
        // for(int i = height.size() - 2; i >= 0; i--) {
        //     maxRight[i] = max(maxRight[i + 1], height[i]);
        // }
        // for(int i = 0; i < height.size(); i++) {
        //     if(i == 0 || i == height.size() - 1) continue;
        //     int h = min(maxLeft[i], maxRight[i]) - height[i];
        //     if(h > 0) sum+=h;
        // }
        // return sum;
        // 解法3，单调栈，维持栈中的元素从栈顶到栈底是一个从小到大的排序
        if(height.size() <= 2)return 0;
        stack<int> st;
        st.push(0);
        int sum = 0;
        for(int i = 0; i <height.size(); i++) {
            if(height[i] < height[st.top()]) {
                st.push(i);
            } else if(height[i] == height[st.top()]) {
                st.pop();
                st.push(i);
            } else { //height[i] > height[st.top()]
                while(!st.empty() && height[i] >height[st.top()]) {
                    int mid = st.top();
                    st.pop();
                    if(!st.empty()) {
                        int h = min(height[st.top()], height[i]) - height[mid];
                        int w = i - st.top() - 1;  //注意减1
                        sum += h * w;
                    }
                }
            } 
            st.push(i);
        }
        return sum;
    }
```



## 3 滑动窗口
### 3.1 leetcode 3 无重复字符的最长子串
![](https://cdn.nlark.com/yuque/0/2024/png/44540266/1730368469543-b5ad3303-5d98-470d-b8e2-e53c9c30f94f.png)

```cpp
int lengthOfLongestSubstring(string s) {
        int maxLen = 0;
        int left = 0; 
        int right = 0;
        unordered_set<char> used;
        while(left < s.size()) {
            unordered_set<char> used;
            right = left;
            while(right < s.size()) {
                if(used.find(s[right]) != used.end()) {
                    break;
                } else {
                    used.insert(s[right]);
                }
            }
            maxLen = max(maxLen, (right - left) + 1);
            left++;
        }
        return maxLen;
    }
```



### 3.2 leetcode 438 找到字符串中所有字母异位词
![](https://cdn.nlark.com/yuque/0/2024/png/44540266/1730516314766-7edec480-d97b-4134-bcc2-3795bc72c47c.png)

```cpp
bool compareVec(vector<int>& nums1, vector<int>& nums2) {
        for(int i = 0; i < nums1.size(); i++) {
            if(nums1[i] != nums2[i]) return false;
        }
        return true;
    }
    vector<int> findAnagrams(string s, string p) {
        vector<int> ans;
        if(s.size() < p.size()) return ans;
        vector<int> charMap(26, 0);
        vector<int> sCharMap(26, 0);
        for(int i = 0; i < p.length(); i++) charMap[p[i] - 'a']++;
        int right = 0;
        int left = 0;
        while(right < p.length()) sCharMap[s[right++] - 'a']++;
        while(right < s.length()) {
            if(compareVec(charMap, sCharMap) == true) ans.push_back(left);
            sCharMap[s[right] - 'a']++;
            sCharMap[s[left] - 'a']--;
            right++;
            left++;
        }
        if(compareVec(sCharMap, charMap) == true) ans.push_back(left);
        return ans;
    }
```

## 4 子串
### 4.1 leetcode 560 和为k的子数组
![](https://cdn.nlark.com/yuque/0/2024/png/44540266/1731049934359-9acd7bc5-880a-4363-a01c-d67306535c6a.png)

```cpp
int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> prefixSumCount;
        int prefixSum = 0;
        int ans = 0;
        prefixSumCount[0] = 1;  //前缀和为0，需要初始化为1，因为当前的前缀和可能为k，此时prefixSumCount[prefixSum - k]=prefixSumCount[0]
        for(int i = 0 ; i < nums.size(); i++) {
            prefixSum += nums[i];
            ans += prefixSumCount[prefixSum - k];
            prefixSumCount[prefixSum]++;
        }
        return ans;
    }
```



### 4.2 leetcode 239 滑动窗口最大值
![](https://cdn.nlark.com/yuque/0/2024/png/44540266/1731079956234-70512a6a-ed74-4f62-a910-cfde28f1b503.png)

```cpp
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> que; //单调队列
        vector<int> ans;
        if(nums.size() == 0) return ans;
        que.push_back(nums[0]);
        int right = 1;
        int left = 0;
        while(right < k) {
            while(!que.empty() && nums[right] > que.back()) {
                que.pop_back();
            }
            que.push_back(nums[right++]);
        }
        ans.push_back(que.front());
        while(right < nums.size()) {
            while(!que.empty() && nums[right] > que.back()) {
                que.pop_back();
            }
            que.push_back(nums[right]);
            if(nums[left] == que.front()) {  //如果弹出的元素为最大值
                que.pop_front();
            }
            ans.push_back(que.front());
            right++;
            left++;
        }
        return ans;
    }
```



### 4.3 leetcode 76 最小覆盖子串
题目描述：
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。\
注意：
对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
如果 s 中存在这样的子串，我们保证它是唯一的答案。
```plain
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
```

解析：使用滑动窗口，用left和right双指针实现

```cpp
bool checkIfMatch(unordered_map<char, int>& sCharMap, unordered_map<char, int>& tCharMap) {
        for(const auto& item: tCharMap) {
            if(sCharMap.find(item.first) == sCharMap.end()) return false;
            if(item.second > sCharMap[item.first]) return false;
        }
        return true; 
    }

    // 最小覆盖子串
    string minWindow(string s, string t) {
        unordered_map<char, int> sCharMap;
        unordered_map<char, int> tCharMap;
        for(const auto& ch: t) {
            tCharMap[ch]++;
        }
        int left = 0;
        int right = 0;
        int minLen = INT_MAX;
        int start = -1;
        while(right < s.length()) {
            if(tCharMap.find(s[right]) != tCharMap.end()) {
                sCharMap[s[right]]++;
            }
            while(checkIfMatch(sCharMap, tCharMap) && left <= right) {
                if(right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    start = left;
                }
                sCharMap[s[left]]--;
                left++;
            }
            right++;
        }
        return minLen == INT_MAX? "": s.substr(start, minLen);
    }
```




## 5 普通数组
### 5.1 leetcode 53 最大子数组和
题目描述：
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
子数组
是数组中的一个连续部分。

```plain
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```

```cpp
    int maxSubArray(vector<int>& nums) {
        int curSum = nums[0];
        int maxSum = nums[0];
        int curIndex = 1;
        while(curIndex < nums.size()) {
            if(curSum + nums[curIndex] > nums[curIndex] ) {
                curSum += nums[curIndex];
            } else {
                curSum = nums[curIndex];
            }
            if(curSum >= maxSum) maxSum = curSum;  //注意要在判断晚curSum + nums[curIndex]与nums[curIndex]和project
 
            curIndex++;
        }
        return maxSum;
    }   
```

### 5.2  leetcode 56 合并区间
题目描述：
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
示例 1: 
输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
题解: 
```C++
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> ans;
        if(intervals.size() == 0) return ans;
        sort(intervals.begin(), intervals.end(), [](const vector<int>& nums1, const vector<int>& nums2){
            if(nums1[0] != nums2[0]) {
                return nums1[0] < nums2[0];
            } else {
                return nums1[1] < nums2[1];
            }
        });
        ans.push_back(intervals[0]);
        int lastRight = intervals[0][1];
        for(int i = 1; i < intervals.size();i++) {
            if(intervals[i][0] > lastRight) {
                ans.push_back(intervals[i]);
                lastRight = intervals[i][1];
            } else {
                lastRight = max(lastRight, intervals[i][1]);
                ans.back()[1] = lastRight;
            }
        }
        return ans;
    }
```

### 5.3 leetcode 189 轮转数组
题目描述：
给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
>示例 1: \
>输入: nums = [1,2,3,4,5,6,7], k = 3 \
>输出: [5,6,7,1,2,3,4] \
>解释: \
>向右轮转 1 步: [7,1,2,3,4,5,6] \
>向右轮转 2 步: [6,7,1,2,3,4,5] \
>向右轮转 3 步: [5,6,7,1,2,3,4] \

解题:
Method 1:
```C++
void rotate(vector<int>& nums, int k) {
    vector<int> ans(nums.size());
    k = k  % nums.size();
    for(int i = 0; i < nums.size();i++) {
        ans[(i + k) % nums.size() ] = nums[i];
    }
    for(int i = 0; i < ans.size(); i++) {
        nums[i] = ans[i];
    }
}
```
Method 2
```C++
    void rotate(vector<int>& nums, int k) {
        int n = nums.size();
        k = k % n;
        int count = gcd(k, n);
        for(int start = 0; start < count; start++) {
            int cur = start;
            int temp = nums[cur];
            do{
                int nextIndext = (cur + k) % n;
                swap(nums[nextIndext], temp);
                cur = nextIndext;
            }while(cur != start);
        }
    }
```
Method 3
```C++
    void reverse(vector<int>& nums, int start, int end) {
        while(start < end) {
            swap(nums[start], nums[end]);
            start++;
            end--;
        }
    }
    void rotate(vector<int>& nums, int k) {
        k = k % nums.size();
        reverse(nums, 0, nums.size() - 1); //翻转整个数组
        reverse(nums, 0, k - 1);  //翻转前k个元素
        reverse(nums, k, nums.size() - 1); //翻转后面的元素
    }
```

### 5.4 leetcode 238 除自身意外数组的乘积
题目描述：
给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。
>请不要使用除法，且在 O(n) 时间复杂度内完成此题。 \
>示例 1: \
>输入: nums = [1,2,3,4] \
>输出: [24,12,8,6]
题解: 三角型累乘
```C++
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> ans(nums.size(), 1);
        for(int i = 1; i < nums.size(); i++) {
            ans[i] = ans[i - 1] * nums[i - 1];
        }
        int temp = nums.back();
        for(int i = nums.size() - 2; i >= 0; i--) {
            ans[i] = ans[i] * temp;
            temp *= nums[i];
        }
        return ans;
    }
```

### 5.5 leetcode 41 缺失的第一个正数
题目描述：
给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。
请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
题解：
```C++
   int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        for(auto& num: nums) {
            if(num <= 0) {
                num = n + 1;
            }
        }
        for(auto& num: nums) {
            if(abs(num) > n) {
                continue;
            } else {
                nums[abs(num) - 1] = - abs(nums[abs(num) - 1]); //关键步骤，将出现过的数对应的下标的数都设为负数
            }
        }
        for(int i = 0; i < nums.size();i++) {
            if(nums[i] > 0) return i + 1;
        } 
        return n + 1;
    }
```
## 6 矩阵
### 6.1 leetcode 73 矩阵置零
题目描述：
给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
示例：
输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
输出：[[1,0,1],[0,0,0],[1,0,1]]

题解：
```C++
    void setZeroes(vector<vector<int>>& matrix) {
        unordered_set<int> usedRows;
        unordered_set<int> usedCols;
        for(int i = 0; i < matrix.size();i++) {
            for(int j = 0; j < matrix[0].size(); j++) {
                if(matrix[i][j] == 0) {
                    if(usedRows.find(i) == usedRows.end()) usedRows.insert(i);
                    if(usedCols.find(j) == usedCols.end()) usedCols.insert(j);
                } else {
                    continue;
                }
            }
        }
        for(int i = 0; i < matrix.size();i++) {
            for(int j = 0; j < matrix[0].size();j++) {
                if(usedRows.find(i) != usedRows.end() || usedCols.find(j) != usedCols.end()) matrix[i][j] = 0;
            }
        }
    }
```

### 6.2 leetcode 54 螺旋矩阵
题目描述
给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
示例 1：
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```C++
    int directs[4][2] = {{0, 1},{1, 0}, {0, -1}, {-1, 0}}; // 四个方向
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<vector<bool>> used(matrix.size(), vector<bool>(matrix[0].size(), false));
        int sum = matrix.size() * matrix[0].size();
        vector<int> ans(sum);
        int curx = 0;
        int cury = 0;
        int count = 1;
        while(count <= sum) {
            for(int i = 0; i < 4; i++) {
                int dx = curx + directs[i][0];
                int dy = cury + directs[i][1];
                while(dx < matrix.size() && dx >=0 && dy < matrix[0].size() && dy >= 0 && used[dx][dy] == false) {
                    ans.push_back(matrix[dx][dy]);
                    count++;
                    used[dx][dy] = true;
                    curx = dx;
                    cury = dy;
                    dx = curx + directs[i][0];
                    dy = cury + directs[i][1];
                }
            }
        }
        return ans;
    }
```
### 6.3 leetcode 48 旋转图像

题目描述
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。 \
你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。 \
示例 1：\
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
C++版
``` C++
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for(int i = 0; i < n/2; i++) {
            for(int j = 0; j < (n + 1) / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n-j-1][i];
                matrix[n-j-1][i] = matrix[n-i-1][n-j-1];
                matrix[n-i-1][n-j-1] = matrix[j][n-i-1];
                matrix[j][n-i-1] = temp;
            }
        }
    }
```
Go语言版
```Go
func rotate(matrix [][]int)  {
    n := len(matrix)
    for i:= 0; i < n/2; i++ {
        for j := 0; j < (n + 1) / 2; j++ {
            matrix[i][j], matrix[n-j-1][i], matrix[n-i-1][n-j-1],matrix[j][n-i-1]=
            matrix[n-j-1][i],matrix[n-i-1][n-j-1],matrix[j][n-i-1],matrix[i][j]
        }
    }
}
```
### 6.4 leetcode 240 搜索二维矩阵||

题目描述
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性： \
每行的元素从左到右升序排列。 \
每列的元素从上到下升序排列。 \
示例 1： \
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5 \
输出：true
题解：
使用二分法，用一层二分即可，两层二分容易找不到第一层起点
```C++
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        for(const auto& row: matrix) {
            // lower_bound()
            auto iter = lower_bound(row.begin(), row.end(), target);   //lower_bound功能：在一个有序范围内查找第一个不小于给定值的位置
            if(iter != row.end() && *iter == target) {
                return true;
            }
        }
        return false;
    }
```

## 7 链表
### 7.1  leetcode 160 相交链表
题目描述：
给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。
图示两个链表在节点 c1 开始相交：
![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)
题解： 
```C++
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(headA == nullptr || headB == nullptr) return nullptr;
        int len1 = 0;
        int len2 = 0;
        ListNode* cur1 = headA;
        ListNode* cur2 = headB;
        while(cur1 != nullptr) {
            len1++;
            cur1 = cur1->next;
        }
        while(cur2 != nullptr) {
            len2++;
            cur2 = cur2->next;
        }
        cur1 = headA;
        cur2 = headB;
        int len = 0;
        if(len1 <= len2) {
            len = len2 - len1;
            while(len--) {
                cur2 = cur2->next;
            }
        } else {
            len = len1 - len2;
            while(len--) {
                cur1 = cur1->next;
            }
        }
        while(cur1 != nullptr && cur2 != nullptr && cur1 !=cur2) {
            cur1 = cur1->next;
            cur2 = cur2->next;
        }
        if(cur1 != nullptr && cur1 == cur2) {
            return cur1;
        } else {
            return nullptr;
        }
    }
```
更简洁的写法：
```C++
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(headA == nullptr || headB == nullptr) return nullptr;
        ListNode* pA = headA;
        ListNode* pB = headB;
        while(pA != pB) {
            pA = pA == nullptr ? headB : pA->next;
            pB = pB == nullptr ? headA : pB->next;   
        }
        return pA;
    }
```
### 7.2 leetcode 206 反转链表
题目描述：
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。 \
示例 1：
![](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)
题解：
```C++
  ListNode* reverseList(ListNode* head){
    if(head == nullptr) return nullptr;
    ListNode* pCur = head;
    ListNode* pPre = nullptr;
    while(pCur != nullptr) {
        ListNode* tmp = pCur->next;
        pCur->next = pPre;
        pPre = pCur;
        pCur = tmp;
    }
    return pPre;
  }
```

### 7.3 leetcode 234 回文链表
题目描述：
给你一个单链表的头节点 head ，请你判断该链表是否为
回文链表。如果是，返回 true ；否则，返回 false 。 \
示例 1： \
![](https://assets.leetcode.com/uploads/2021/03/03/pal1linked-list.jpg)
输入：head = [1,2,2,1] \
输出：true \
题解： 使用递归方式来进行判断(用数组接收节点值来判断的方法暂不写)
```C++
    ListNode* frontPointer;
    bool recursivelyCheck(ListNode* currentNode) {
        if(currentNode != nullptr) {
            if(!recursivelyCheck(currentNode->next)) {
                return false;
            }
            if(currentNode->val != frontPointer->val) {
                return false;
            }
            frontPointer = frontPointer->next;
        }
        return true;
    }
    bool isPalindrome(ListNode* head) {
        frontPointer = head;
        return recursivelyCheck(head);
    }
```

### 7.4 leetcode 141 环形链表
题目描述：
给你一个链表的头节点 head ，判断链表中是否有环。
如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。
如果链表中存在环 ，则返回 true 。 否则，返回 false 。 \
示例 1： \
![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)
题解： 快慢双指针法
```C++
    bool hasCycle(ListNode *head) {
        if(head == nullptr) return false;
        if(head->next == nullptr) return false;
        // 使用快慢指针
        ListNode* slow = head;
        ListNode* fast = head->next;
        while(fast != slow  && fast->next != nullptr && fast->next->next != nullptr) {
            fast = fast->next->next;
            slow = slow->next;
        }
        if(fast == slow && fast != nullptr) {
            return true;
        } 
        return false;
    }
```

### 7.5 leetcode 142 环形链表||
题目描述: \
给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
不允许修改 链表。
示例1：
![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)
题解： \
快慢指针进阶(https://www.programmercarl.com/0142.%E7%8E%AF%E5%BD%A2%E9%93%BE%E8%A1%A8II.html#%E7%AE%97%E6%B3%95%E5%85%AC%E5%BC%80%E8%AF%BE)
```C++ 
    ListNode *detectCycle(ListNode *head) {
        // 快慢双指针
        ListNode* fast = head;
        ListNode* slow = head;
        while(fast != nullptr && fast->next != nullptr) {
            fast = fast->next->next;
            slow = slow->next;
            if(fast == slow) {
                fast = head;
                while(fast != slow ) {  // x = z
                    fast = fast->next;
                    slow = slow->next;
                }
                return fast;
            } 
        }
        return nullptr;
    }
```

### 7.6 leetcode  合并两个有序链表
题目描述： \
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 \
示例1:
![](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)
题解： \
``` C++
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* dummyNode = new ListNode(-1);
        ListNode* pCur = dummyNode;
        ListNode* pL1 = list1;
        ListNode* pL2 = list2;
        while(pL1 != nullptr && pL2 != nullptr) {
            if(pL1->val < pL2->val) {
                pCur->next = pL1;
                pCur = pL1;
                pL1 = pL1->next;
            } else {
                pCur->next = pL2;
                pCur = pL2;
                pL2 = pL2->next; 
            }
        }
        if(pL1 != nullptr) {
            pCur->next = pL1;
        } else {
            pCur->next = pL2;
        }
        return dummyNode->next;
    }
```
### 7.7 leetcode 2 两数相加
题目描述: \
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
请你将两个数相加，并以相同形式返回一个表示和的链表。 
你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/01/02/addtwonumber1.jpg)
示例1： \
输入：l1 = [2,4,3], l2 = [5,6,4] \
输出：[7,0,8] \
解释：342 + 465 = 807. \
```C++
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) { 
        if(l1 == nullptr) return l2;
        if(l2 == nullptr) return l1;
        int sum = 0;
        ListNode* dummyNode = new ListNode(-1);
        ListNode* pCur = dummyNode;
        ListNode* pL1 = l1;
        ListNode* pL2 = l2;
        while(pL1 != nullptr && pL2 != nullptr) {
            int tempSum = pL1->val + pL2->val + sum;
            sum = tempSum / 10;
            tempSum = tempSum % 10;
            ListNode* newNode = new ListNode(tempSum);
            pCur->next = newNode;
            pCur = newNode;
            pL1 = pL1->next;
            pL2 = pL2->next;
        }
        ListNode* pList = pL1 == nullptr ? pL2 : pL1;
        while(pList != nullptr) {
            int tempSum = pList->val + sum;
            sum = tempSum / 10;
            tempSum = tempSum % 10;
            ListNode* newNode = new ListNode(tempSum);
            pCur->next = newNode;
            pCur = newNode;
            pList = pList->next;
        } 
        if(sum != 0) {
            ListNode* lastNode = new ListNode(sum);
            pCur->next = lastNode;
        }
        return dummyNode->next;
    }
```

### 7.8 leetcode 19 删除链表的第n个节点
题目描述： \
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。 \
示例1： \
![](https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg)
输入：head = [1,2,3,4,5], n = 2 \
输出：[1,2,3,5] \
题解： 滑动窗口，维持一个内部有n个节点的窗口，一直滑动到末尾删除左侧节点即可\
```C++
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummyNode = new ListNode(-1);
        dummyNode->next = head;
        ListNode* pRight = dummyNode;
        int count = 0;
        while(count < n && pRight != nullptr) {
            pRight = pRight->next;
            count++;
        }
        if(pRight == nullptr) {
            return nullptr;
        }
        ListNode* pLeft = dummyNode;
        while(pRight->next != nullptr) {
             pRight = pRight->next;
             pLeft = pLeft->next;
        }
        pLeft->next = pLeft->next->next;
        return dummyNode->next;
    }
```
### 7.9 leetcode 24 两两交换链表中的节点
题目描述: \
给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。 \
示例1： \
![](https://assets.leetcode.com/uploads/2020/10/03/swap_ex1.jpg)
输入：head = [1,2,3,4] \
输出：[2,1,4,3] \
题解： 节点间的交换，最好画图表示不然容易逻辑混乱\
```C++
    ListNode* swapPairs(ListNode* head) {
        if(head == nullptr || head->next == nullptr) return head;
        ListNode* dummyNode = new ListNode(-1);
        dummyNode->next = head;
        ListNode* pRight = dummyNode;
        ListNode* pLeft = dummyNode;
        while(pRight->next != nullptr) {
            pRight = pRight->next;
            if(pRight->next == nullptr) break;
            pRight = pRight->next;
            pLeft->next->next = pRight->next;  //先将后续节点保存好
            pRight->next = pLeft->next;
            pLeft->next = pRight;
            pLeft = pLeft->next->next;
            pRight = pLeft;
        }
        return dummyNode->next;
    }
```

### 7.10 leetcode 25 k个一组反转链表
题目描述: \
给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
示例1： \
![](https://assets.leetcode.com/uploads/2020/10/03/reverse_ex1.jpg)
输入：head = [1,2,3,4,5], k = 2 \
输出：[2,1,4,3,5] \
题解: \
```C++
  ListNode* reverseKGroup(ListNode* head, int k){
        if(head == nullptr) return nullptr;
        ListNode* dummyNode = new ListNode(-1, head);  //使用辅助节点统一情况
        ListNode* fast = dummyNode;
        ListNode* slow = dummyNode;
        while(true) {
            for(int i = 0; i < k && fast != nullptr; i++) {
                fast = fast->next;
            }
            if(fast == nullptr) break;              //不足k个一组
            ListNode* nextStartPtr = slow->next;    //保存下一次迭代位置
            ListNode* pre = fast->next;             //本次操作前置节点
            fast = slow->next;                      //本次操作第一个操作节点
            for(int i = 0; i < k; i++) {
                ListNode* tmp = fast->next;
                fast->next = pre;
                pre = fast;
                fast = tmp;
            }
            slow->next = pre;  // 连接上上一段
            slow = nextStartPtr;  //更新起始位置
            fast = nextStartPtr;
        }
        return dummyNode->next;
  }
```

### 7.11 leetcode 138 随机链表的复制
题目描述: \
给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。 \
例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。
返回复制链表的头节点。
用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示： \
val：一个表示 Node.val 的整数。 \
random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。 \
你的代码只接受原链表的头节点 head 作为传入参数。 \
示例1 ：\
![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e1.png)
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]] \
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]] \
题解: \
```C++
    unordered_map<Node*, Node*> cacheNode;  //使用哈希表记录<节点旧地址，节点新地址>
    Node* copyRandomList(Node* head) {
        if(head == nullptr) return nullptr;
        if(!cacheNode.count(head)) {  //如果碰到没有保存的节点
            Node* headNew = new Node(head->val);
            cacheNode[head] = headNew;
            headNew->next = copyRandomList(head->next);
            headNext->random = copyRandomList(head->random);
        }
        return cacheNode[head];
    }
```

### 7.12 leetcode 148 排序链表
题目描述: \
给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。\
示例 1：\
![](https://assets.leetcode.com/uploads/2020/09/14/sort_list_1.jpg)
输入：head = [4,2,1,3] \
输出：[1,2,3,4] \
题解： 
朴素解法，冒泡
```C++
    ListNode* sortList(ListNode* head) {
        if(head == nullptr) return head;
        bool flag = true;
        ListNode* dummyNode = new ListNode(-1, head);
        while(true) {
            // 开始一趟排序
            ListNode* curNode = dummyNode->next;
            ListNode* pre = dummyNode;
            flag = true;
            while(curNode->next != nullptr) {
                if(curNode->next->val < curNode->val) {
                    //进行交换
                    ListNode* tmp = curNode->next->next;
                    pre->next = curNode->next;
                    curNode->next->next = curNode;
                    curNode->next = tmp;
                    pre = pre->next;
                    curNode = pre->next;
                    flag = false;
                } else {
                    pre = pre->next;
                    curNode = curNode->next;
                }
            }
            if(flag) {
                break;
            }
        }
        return dummyNode->next;
    }
```
优化版本，使用归并排序算法
```C++
    ListNode* sortList(ListNode* head) {
        if(head == nullptr || head->next == nullptr) return head;
        //找到链表中点
        ListNode* mid = getMid(head);
        ListNode* left = head;
        ListNode* right = mid->next;
        mid->next = nullptr;    // 先断开链表
        left = sortList(left);  // 排序左边链表
        right = sortList(right); // 排序右边链表
        return merge(left, right);
    }
    // **找到链表中点（快慢指针法）**
    ListNode* getMid(ListNode* head) {
        ListNode* slow = head;
        ListNode* fast = head->next;
        while(fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }
    // **合并两个有序链表**
    ListNode* merge(ListNode* head1, ListNode* head2) {
        ListNode* dummyHead = new ListNode(-1);
        ListNode* cur = dummyHead;
        ListNode* temp1 = head1;
        ListNode* temp2 = head2;
        while(temp1 != nullptr && temp2 != nullptr) {
            if(temp1->val < temp2->val) {
                cur->next = temp1;
                temp1 = temp1->next;
            } else {
                cur->next = temp2;
                temp2 = temp2->next;
            }
            cur = cur->next;
        }
        if(temp1 != nullptr) {
            cur->next = temp1;
        } else {
            cur->next = temp2;
        }
        return dummyHead->next;
    }
```
时间复杂度：O(N log N)
空间复杂度：O(log N)（递归栈）
稳定排序（不会改变相同元素的相对顺序）


### 7.13 leetcode 23 合并k个升序链表
题目描述： \
给你一个链表数组，每个链表都已经按升序排列。
请你将所有链表合并到一个升序链表中，返回合并后的链表。
示例1：\
输入：lists = [[1,4,5],[1,3,4],[2,6]] \
输出：[1,1,2,3,4,4,5,6]\
解释：链表数组如下：\
[\
  1->4->5,\
  1->3->4,\
  2->6\
]\
将它们合并到一个有序链表中得到。\
1->1->2->3->4->4->5->6\
题解：
解法1 顺序合并，每次都将当前链表合并进结果链表中
```C++
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if(lists.size() == 0 ) return nullptr;
        ListNode* ansHead = nullptr;
        for(int i = 0 ; i < lists.size(); i++) {
            ansHead = merge2Lists(ansHead, lists[i]);
        }
        return ansHead;
    }
    ListNode* merge2Lists(ListNode* head1, ListNode* head2) {
        if((!head1) || (!head2)) return head1 ? head1: head2;
        ListNode* dummyNode = new ListNode(-1);
        ListNode* cur = dummyNode;
        ListNode* temp1 = head1;
        ListNode* temp2 = head2;
        while(temp1 && temp2) {
            if(temp1->val < temp2->val) {
                cur->next = temp1;
                temp1 = temp1->next;
            } else {
                cur->next = temp2;
                temp2 = temp2->next;
            }
            cur = cur->next;
        }
        if(temp1) {
            cur->next = temp1;
        } else {
            cur->next = temp2;
        }
        return dummyNode->next;
    }
```
解法2，使用归并分治思路
```C++
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        return merge(lists, 0, lists.size() - 1);
    }
    ListNode* merge(vector<ListNode*>& lists, int left, int right) {
        if(left == right) return lists[left];
        if(left > right) return nullptr;
        int mid = (left + right) >> 1; // 除以2
        return merge2Lists(merge(lists, left, mid), merge(lists, mid + 1, right));
    }

    ListNode* merge2Lists(ListNode* head1, ListNode* head2) {
        if((!head1) || (!head2)) return head1 ? head1: head2;
        ListNode* dummyNode = new ListNode(-1);
        ListNode* cur = dummyNode;
        ListNode* temp1 = head1;
        ListNode* temp2 = head2;
        while(temp1 && temp2) {
            if(temp1->val < temp2->val) {
                cur->next = temp1;
                temp1 = temp1->next;
            } else {
                cur->next = temp2;
                temp2 = temp2->next;
            }
            cur = cur->next;
        }
        if(temp1) {
            cur->next = temp1;
        } else {
            cur->next = temp2;
        }
        return dummyNode->next;
    }
```

### 7.14 leetcode 146 LRU缓存
题目描述: \
请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。\
实现 LRUCache 类：\
LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存\
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。\
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。\
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。 \
示例: \
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]\
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]\
输出\
[null, null, null, 1, null, -1, null, -1, 3, 4]\
解释：\
LRUCache lRUCache = new LRUCache(2);\
lRUCache.put(1, 1); // 缓存是 {1=1}\
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}\
lRUCache.get(1);    // 返回 1\
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}\
lRUCache.get(2);    // 返回 -1 (未找到)\
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}\
lRUCache.get(1);    // 返回 -1 (未找到)\
lRUCache.get(3);    // 返回 3\
lRUCache.get(4);    // 返回 4\
题解：
```C++
class LRUCache {
public:
    LRUCache(int capacity) {
        m_capacity = capacity;
    }
    
    int get(int key) {
        if(m_hashTable.count(key) == 0) return -1;
        auto iter = m_hashTable[key];
        m_items.splice(m_items.begin(), m_items, iter);  // 将指定元素移动到最前面，注意std::list中方法splice的使用
        return iter->second;
    }
    
    void put(int key, int value) {
        if(m_hashTable.find(key) != m_hashTable.end()) {
            auto iter = m_hashTable[key];
            iter->second = value;
            m_items.splice(m_items.begin(), m_items, iter);
            return;
        }
        if(m_items.size() >= m_capacity) {  //如果已经达到容量上限，则删除最后一个
            auto delKey = m_items.back().first;
            m_hashTable.erase(delKey);
            m_items.pop_back();
        }
        m_items.push_front({key, value});
        m_hashTable[key] = m_items.begin();
    }
private:
    unordered_map<int, list<pair<int, int>>::iterator> m_hashTable;  // <key, iter>
    list<pair<int, int>> m_items;
    size_t m_capacity;
};
```



## 8 二叉树
### 8.1 leetcode 94 二叉树的中序遍历
题目描述: \
给定一个二叉树的根节点 root，返回 它的中序遍历。\
示例1：
![](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)
输入：root = [1,null,2,3]\
输出：[1,3,2]\
题解：
方法1 直接递归法\
```cpp
    vector<int> ans;
    void inOrderedTrace(TreeNode* root) {
        if(root == nullptr) return;
        inOrderedTrace(root->left);
        ans.push_back(root->val);
        inOrderedTrace(root->right);
    }
    vector<int> inorderTraversal(TreeNode* root) {
        inOrderedTrace(root);
        return ans;
    }
```
方法2 非递归法
```cpp
vector<int> inOrderedTrace(TreeNode* root) {
    TreeNode* cur = root;
    stack<TreeNode*> st;
    vector<int> vec;
    while(cur != nullptr || !st.empty()){
        if(cur != nullptr) {  // 如果还没有遍历到叶子节点，则继续遍历
            st.push(cur);
            cur = cur->left;
        } else {
            cur = st.top();
            st.pop();
            vec.push_back(cur->val);
            cur = cur->right;
        }
    }
    return vec;
}

```

### 8.2 leetcode 104 二叉树的最大深度
题目描述：\
给定一个二叉树 root ，返回其最大深度。
二叉树的 最大深度 是指从根节点到最远叶子节点的最长路径上的节点数。
示例1：
![](https://assets.leetcode.com/uploads/2020/11/26/tmp-tree.jpg)
输入：root = [3,9,20,null,null,15,7]\
输出：3
题解：直接使用前序遍历即可
```cpp
    int preorderTree(TreeNode* root, int depth) {
        if(root == nullptr) return depth;
        return max(preorderTree(root->left, depth + 1), preorderTree(root->right, depth + 1));
    } 
    int maxDepth(TreeNode* root) {
        return preorderTree(root, 0);
    }
```

### 8.3 leetcode 226 翻转二叉树
题目描述:\
给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。
示例1：
![](https://assets.leetcode.com/uploads/2021/03/14/invert1-tree.jpg)
输入：root = [4,2,7,1,3,6,9]\
输出：[4,7,2,9,6,3,1]
题解：直接从上往下递归反转即可
```cpp
    TreeNode* invertTree(TreeNode* root) {
        if(root == nullptr) return root;
        swap(root->left, root->right);
        invertTree(root->left);
        invertTree(root->right);
        return root;
    }
```

### 8.4 leetcode 101 对称二叉树
题目描述：\
给你一个二叉树的根节点 root ， 检查它是否轴对称。\
示例1：\
![](https://pic.leetcode.cn/1698026966-JDYPDU-image.png)
输入：root = [1,2,2,3,4,4,3]\
输出：true\
题解：\
使用迭代方式，针对左子树和右子树对称的位置判断节点值是否相等
```cpp
    bool isSymmetricTree(TreeNode* leftTree, TreeNode* rightTree) {
        if(leftTree == nullptr && rightTree == nullptr) return true;
        if((leftTree == nullptr && rightTree != nullptr) || (leftTree != nullptr && rightTree == nullptr) || leftTree->val != rightTree->val) return false;
        bool left = isSymmetricTree(leftTree->left, rightTree->right);
        bool right = isSymmetricTree(leftTree->right, rightTree->left);
        if(left == false || right == false) return false;
        return true;
    } 
    bool isSymmetric(TreeNode* root) {
        if(root == nullptr) return false;
        if(root->left == nullptr && root->right == nullptr) return true;
        return isSymmetricTree(root->left, root->right);
    }
```
### 8.5 leetcode 543 二叉树的直径
题目描述：\
给你一棵二叉树的根节点，返回该树的 直径 。
二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。
两节点之间路径的 长度 由它们之间边数表示。\
示例1：\
![](https://assets.leetcode.com/uploads/2021/03/06/diamtree.jpg)
输入：root = [1,2,3,4,5]\
输出：3\
解释：3 ，取路径 [4,2,1,3] 或 [5,2,1,3] 的长度。\
题解：\
```cpp

```







## 9 图论








## 10 回溯






## 11 二分查找










## 12 栈




## 13 贪心算法






## 14 动态规划




## 15 多维动态规划






## 16 技巧












