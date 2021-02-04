# screen_detect

1. 路径

```cpp
├─logo     // 存放需要检测的 logo 图片
├─output   // 输出文件夹
│  ├─mask_all   // 每个文件夹代表一个检测的截图
│  │  ├─LOGO    // 对 logo 的检测结果图片
│  │  └─PATTERN // 以下是对每个 menu 的检测结果图片
│  │      ├─menu_bangzhu
│  │      ├─menu_bianji
│  │      ├─menu_chakan
│  │      ├─menu_geshi
│  │      ├─menu_wenjian
│  │      └─x
│  ├─no_tubiao
│  ├─no_wenjian
│  ├─raw
│  └─raw_long
│  └─result.csv // 存放所有图片的所有检测结果
├─pattern  // 存放需要检测的 pattern 图片
└─test_img // 存放待检测的截图
```

2. 把需要检测的截图放入 `test_img` 文件夹中，在该文件夹命令行输入 `run` 运行，在 `output` 内查看结果。
3. 由于 `x` 按钮的图片结构过于简单，特征过少，精确度相对其他按钮稍低。
4. 参考：https://blog.csdn.net/lovebyz/article/details/84999282