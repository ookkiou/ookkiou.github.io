---
title: Python基础
date: 2026-08-07 16:25:15
categories: 算法
---

### str字符串是不能修改的 List列表可以修改 T3794
**str 在 Python 里就是"只读的列表"**：可以读、可以切片、可以拼成新串，但**任何想"改一个字符"的操作都会被拒**。
#### 方法一：str转成list
lst = list(s) 
对lst做修改
return ''.join(lst)  返回str
#### 方法二：切片
`class Solution:`
    `def reversePrefix(self, s: str, k: int) -> str:`
        `return s[:k][::-1] + s[k:]`

- join前面加的东西，代表了连接符是什么
"-".join(["a", "b", "c"])       # "a-b-c"
"".join(["a", "b", "c"])        # "abc"        ← 没有连接符
"  ".join(["a", "b", "c"])      # "a  b  c"    ← 双空格
", ".join(["a", "b", "c"])      # "a, b, c"    ← 逗号空格
"\n".join(["a", "b", "c"])      # "a\nb\nc"    ← 换行（多行字符串）


字符串反转可以用：reverse = word[::-1]
List反转是：lst[0:k] = reversed(lst[0:k]) 就是翻转0到K的字符。