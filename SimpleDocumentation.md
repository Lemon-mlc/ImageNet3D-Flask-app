使用说明
安装包
启动前需安装：Flask、numpy、matplotlib、pillow、opencv-python-headless、flask-login。可一次性安装：pip install Flask numpy matplotlib pillow opencv-python-headless flask-login

程序运行时会在当前目录创建imagenet3dstorage文件夹，其下一层有空sqlite文件，将其内容替换为软件根目录下的database_default.sqlite即可。若不懂替换，可用Navicat Premium软件。

文件结构如下：

        
## Repository Structure
'''
📁 ImageNet3D-Flask-app/
├── 📁 static/
│   ├── 📁 images/
│   ├── 📁 tmp/
├── 📁 templates/
│   ├── login.html
│   ├── account.html
│   ├── test.html
│   ├── index.html
│   ├── eval.html
│   ├── quality.html
│   ├── doc.html
├── app.py
├── config.py
├── render.py
├── database.sqlite
├── database_default.sqlite
├── README.md
├── README.zh.md
├── requirements.txt
└── 📁 logs/
'''
<br>

    
概述
之前研究了带伪3D标注的合成图像，约75%符合标注，其余有视角或形状问题。虽数据集有噪声，但NMMs能学习并提升性能。这为3D理解带来机会，但需准确3D标注数据集评估模型。本项目通过一些参数描述物体，还关注子类型，每张真实图像有预训练注释，需完善预测和标注质量，开始前要读教程。

v2更新（2024年3月）
新增粗调按钮，因处理难类别时3D方向初始设置不佳。
标注可能更耗时，要耐心等待且确保视觉正确。
图片质量可能低，对“不良”对象要标注并忽略，可跳过其他参数调整。
标注可见性和场景密度仍重要。
类别注意事项
双人自行车：用普通自行车模型，保证2D中心和3D方向匹配。
烧杯：按缺口方向匹配3D视点，其他参数简单。
双簧管：忽略“方位角”，匹配其他参数。
拐杖：只标注腋下拐杖类型，其他标为“质量差/无对象”。
沙袋：只标注拳击袋和悬挂袋（圆柱形），忽略“方位角”。
用户界面
用ID登录，看到任务列表和进度。点击任务跳转至未标注问题，顶部有信息和按钮，可切换问题、撤销和保存（按Enter也可保存）。底部有CAD模型列表，除模型参数外，还要回答质量和场景密度问题。保存成功有提示且状态亮起。

指南
匹配网格模型：从列表选最佳匹配模型，很重要，先做。点击按钮可换模型。
3D旋转：用三个参数调整，使渲染对象3D旋转与图像对象对齐，不是对齐分割或边界。
二维位置：确定对象中心位置。
距离：标注时让渲染物体大小与图像物体大小相近。
对象质量：分“良好”“部分可见”“几乎不可见”“质量差/无物体”。
密集场景：分“非密集”和“密集”，根据同类物体在二维平面距离判断。

示例
看教程。

参考文献
[1] 扩散模型中添加3D几何控制
[2] 基于神经特征粗到细渲染的6D位姿估计