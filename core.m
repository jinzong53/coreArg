clear,clc;

%第一步
% 读取图像
img = imread('test1.png');
% 创建滤波器
W = fspecial('gaussian',[5,5],1);  %5*5的
img_Smooth = imfilter(img, W, 'replicate');

%第二步
I=im2double(img_Smooth);
[m,n,c]=size(I);
A=zeros(m,n,c);
%分别处理R、G、B
%先对R进行处理
for i=2:m-1
    for j=2:n-1
        A(i,j,1)=I(i+1,j,1)+I(i-1,j,1)+I(i,j+1,1)+I(i,j-1,1)-4*I(i,j,1);
    end
end
%再对G进行处理
for i=2:m-1
    for j=2:n-1
        A(i,j,2)=I(i+1,j,2)+I(i-1,j,2)+I(i,j+1,2)+I(i,j-1,2)-4*I(i,j,2);
    end
end
%最后对B进行处理
for i=2:m-1
    for j=2:n-1
        A(i,j,3)=I(i+1,j,3)+I(i-1,j,3)+I(i,j+1,3)+I(i,j-1,3)-4*I(i,j,3);
    end
end
B=I-A;

%第三步,
img=rgb2gray(B);
[m,n]=size(img);
BW1=edge(img,'sobel'); %用Sobel算子进行边缘检测,考虑到这些由于不需要跟细微的特征
%BW1=edge(img,'canny'); %用Canny算子进行边缘检测
BW1=im2double(BW1);
f = graythresh(BW1);%查询最佳阈值
BW1 = im2bw(BW1,f); %二值化处理

%第四步，膨胀-腐蚀
%B为膨胀板子
B=[0 1 0
1 1 1
0 1 0];
BW2=imdilate(BW1,B);%图像A1被结构元素B膨胀,我们只选择了一次膨胀
%disk(5)的腐蚀板
se1=strel('disk',5);%这里是创建一个半径为5的平坦型圆盘结构元素
BW3=imerode(BW1,se1);
BW4 = BW2 -BW3;

%图像细化
BW4 = bwmorph(BW4,'thin',5); %Inf表示无法再迭代
figure(1),imshow(BW4);title('原图');hold on
%拟合
%确定拟合对比的变量,由于目前实验次数较少，我们只选择了中心的的偏移程度进行比较，对于相似程度我们选择多种方式进行比较
cx = [];
cy = [];

figure(2),imshow(BW4);title('边缘检测');
[M,N] = size(BW4);
[L,num] = bwlabel(BW4);                           %标签
for i = 50:88 %通过调整只选取了50-100之间的特征点进行拟合，使用过多的标志点进行拟合
    [row,col] = find(L == i);
    conicP = zeros(length(row),2);
    conicP(:,1) = col;
    conicP(:,2) = row;
    figure(1),plot(conicP(:,1)', conicP(:,2)', 'xr');     %drawing sample points
%% 自定义椭圆函数拟合
    a0 = [1 1 1 1 1 1];
    f = @(a,x)a(1)*x(:,1).^2+a(2)*x(:,2).^2+a(3)*x(:,1).*x(:,2)+a(4)*x(:,1)+a(5)*x(:,2)+a(6);%建立方程
    p = nlinfit(conicP , zeros(size(conicP, 1), 1), f,[1 2 3 4 5 6]);
    syms x y
    conic = p(1)*x^2+p(2)*y^2+p(3)*x*y+p(4)*x+p(5)*y+p(6);
    tx = (p(3)*p(5) - 2*p(2)*p(4))/(4*p(1)*p(2)-p(3)*p(3));
    ty = (p(3)*p(5) - 2*p(1)*p(5))/(4*p(1)*p(2)-p(3)*p(3));
    cx = [cx tx];
    cy = [cy ty];
%% 在原图上显示拟合结果
    c = ezplot(conic,[0,N],[0,M]);
    figure(2),set(c, 'Color', 'Blue','LineWidth',2);
end
cx = cx;
cy = cy.';
Standard = cy * cx % 生成40*40的标志点椭圆中心矩阵
%corr2(A,B)


