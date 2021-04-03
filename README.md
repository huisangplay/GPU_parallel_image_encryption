# GPU_parallel_image_encryption
论文加密算法使用的实验环境为：

硬件环境：i7-8750H+gtx1060+8G RAM 操作系统：rhel8  运行环境：opencv4.2+cuda10.2+gcc8.3

代码说明： 代码中在头文件已做了充足的注释，每个文件的作用大致如下： 
1.main.cu为主函数入口，该文件29行的imgUrl的值为本地环境中的图片位置，51行imwrite为存储加密后的图片的位置，54行变量decryption_img为解密后的图像。
 2.analysis作用为：（1）卡方检验（2）将加密后的图片以二进制形式写入文件，方便做NIST测试 
3.chaos：计算二维哈农sine映射和logistic混沌系统的密钥序列 
4.compute：（1）更新密钥（2）对图像执行置乱运算以及置乱的逆运算 
5.dna：dna编码、解码、异或运算、加运算
6.initKey：存储密钥值 
7.mykernel：GPU并行加密实现函数 
8.sha256：计算明文图像的sha-256哈希值
