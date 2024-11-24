## 哈希
###  leetcode 1两数之和
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



### leetcode 49字母异位词分组
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



### leetcode 128最长连续序列
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



## 双指针
### leetcode 283移动零
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

### leetcode 11 盛水最多的容器
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



###  leetcode 15 三数之和
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



### leetcode 42 接雨水
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



## 滑动窗口
### leetcode 3 无重复字符的最长子串
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



### leetcode 438 找到字符串中所有字母异位词
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

## 子串
### leetcode 560 和为k的子数组
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



### leetcode 239 滑动窗口最大值
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



### leetcode 76 最小覆盖子串
题目描述：

<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);">给你一个字符串 </font>`<font style="color:rgba(38, 38, 38, 0.75);background-color:rgb(240, 240, 240);">s</font>`<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> 、一个字符串 </font>`<font style="color:rgba(38, 38, 38, 0.75);background-color:rgb(240, 240, 240);">t</font>`<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> 。返回 </font>`<font style="color:rgba(38, 38, 38, 0.75);background-color:rgb(240, 240, 240);">s</font>`<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> 中涵盖 </font>`<font style="color:rgba(38, 38, 38, 0.75);background-color:rgb(240, 240, 240);">t</font>`<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> 所有字符的最小子串。如果 </font>`<font style="color:rgba(38, 38, 38, 0.75);background-color:rgb(240, 240, 240);">s</font>`<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> 中不存在涵盖 </font>`<font style="color:rgba(38, 38, 38, 0.75);background-color:rgb(240, 240, 240);">t</font>`<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> 所有字符的子串，则返回空字符串 </font>`<font style="color:rgba(38, 38, 38, 0.75);background-color:rgb(240, 240, 240);">""</font>`<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> 。</font>

**<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);">注意：</font>**

+ <font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);">对于</font><font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> </font>`<font style="color:rgba(38, 38, 38, 0.75);background-color:rgb(240, 240, 240);">t</font>`<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> </font><font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);">中重复字符，我们寻找的子字符串中该字符数量必须不少于</font><font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> </font>`<font style="color:rgba(38, 38, 38, 0.75);background-color:rgb(240, 240, 240);">t</font>`<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> </font><font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);">中该字符数量。</font>
+ <font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);">如果 </font>`<font style="color:rgba(38, 38, 38, 0.75);background-color:rgb(240, 240, 240);">s</font>`<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> 中存在这样的子串，我们保证它是唯一的答案。</font>

**<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);">示例 1：</font>**

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





## 普通数组
### leetcode 53 最大子数组和
题目描述：

<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);">给你一个整数数组</font><font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> </font>`<font style="color:rgba(38, 38, 38, 0.75);background-color:rgb(240, 240, 240);">nums</font>`<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);"> </font><font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);">，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。</font>

**<font style="background-color:rgb(240, 240, 240);">子数组</font>**<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);">是数组中的一个连续部分。</font>

**<font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);">示例 1：</font>**

```plain
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```

















## 矩阵








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












