
X = [
    2.9500 6.6300 0
    4.5300 2.7900 0
    3.5700 5.6500 0
    3.1600 5.4700 0
    2.5800 6.4600 1
    2.1600 6.2200 1
    3.2700 3.5200 1
];

X1 = X(X(:,3)==0, 1:2);
X2 = X(X(:,3)==1, 1:2);

hold on
plot(X1(:,1),X1(:,2),'r+','markerfacecolor', [ 1, 0, 0 ]);
plot(X2(:,1),X2(:,2),'b*','markerfacecolor', [ 0, 0, 1 ]);
grid on

M1 = mean(X1);
M2 = mean(X2);
M = mean([X1;X2]);
%第二步：求类内散度矩阵
p = size(X1,1);
q = size(X2,1);
a=repmat(M1,4,1);
S1=(X1-a)'*(X1-a);
b=repmat(M2,3,1);
S2=(X2-b)'*(X2-b);
Sw=(p*S1+q*S2)/(p+q);
%第三步：求类间散度矩阵
sb1=(M1-M)'*(M1-M);
sb2=(M2-M)'*(M2-M);
Sb=(p*sb1+q*sb2)/(p+q);
bb=det(Sw);
%第四步：求最大特征值和特征向量
[V,L]=eig(inv(Sw)*Sb);
[a,b]=max(max(L));
W = V(:,b);%最大特征值所对应的特征向量
%第五步：画出投影线
k=W(2)/W(1);
b=0;
x=2:6;
yy=k*x+b;
plot(x,yy);%画出投影线

%计算第一类样本在直线上的投影点
xi=[];
for i=1:p
    y0=X1(i,2);
    x0=X1(i,1);
    x1=(k*(y0-b)+x0)/(k^2+1);
    xi=[xi;x1];
end
yi=k*xi+b;
XX1=[xi yi];
%计算第二类样本在直线上的投影点
xj=[];
for i=1:q
    y0=X2(i,2);
    x0=X2(i,1);
    x1=(k*(y0-b)+x0)/(k^2+1);
    xj=[xj;x1];
end

yj=k*xj+b;
XX2=[xj yj];
% y=W'*[X1;X2]';
plot(XX1(:,1),XX1(:,2),'r+','markerfacecolor', [ 1, 0, 0 ]);
plot(XX2(:,1),XX2(:,2),'b*','markerfacecolor', [ 0, 0, 1 ]);


w=lda(X(:, 1:2), X(:, 3),unique(X(:, 3)),1);
Y=zeros(size(X,1), 2);
for i=1:size(X,1)
    Y(i,1:2)=X(i,1:2)*w;
end

plot(zeros(p,1),Y(X(:,3)==0,2),'r+','markerfacecolor', [ 0, 0, 1 ]);
plot(zeros(q,1),Y(X(:,3)==1,2),'b*','markerfacecolor', [ 0, 0, 1 ]);
% 只要他们能被一条线分开，就可以使用LDA算法
% 抽取的特征在高维度
