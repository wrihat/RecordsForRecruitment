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

```C++


```















## 链表










## 二叉树










## 图论








## 回溯






## 二分查找










## 栈




## 贪心算法






## 动态规划




## 多维动态规划






## 技巧












