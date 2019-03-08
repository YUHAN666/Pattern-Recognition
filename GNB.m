clear all

%load the data
load spamData;
%estimation of the label prior using ML
lamML = sum(ytrain(:)==1)/numel(ytrain); %lamda ML

%%%%%%%This part is to log transform the features %%%%%%
for i = 1:numel(Xtrain)
    Xtrain(i) = log(Xtrain(i)+0.1);
end
for i = 1:numel(Xtest)
    Xtest(i) = log(Xtest(i)+0.1);
end

%%%%%%% This part is to divide the email into two classes %%%%%%%%%%%
%classOne saves the index of the not spam emails; 
%classTwo saves the index of spam emails
classOne = [];
classTwo = [];
for i = 1:numel(ytrain)
    if ytrain(i) == 0
        classOne = [classOne i];    
    else
        classTwo = [classTwo i];
    end
end

%%%%%This part is to use the ML to calculate the mean and variance of the features%%%
%u(1,j) saves the mean of the jth feature in classOne
%xita(1,j) saves the variance of the jth feature in classOne
%u(2,j) saves the mean of the jth feature in classTwo
%xita(2,j) saves the variance of the jth feature in classTwo

[row, line] = size(Xtrain);
u = zeros(2,line);
xita = zeros(2,line);
for j = 1:line
    for i = 1:numel(classOne)
        u(1,j) = Xtrain(classOne(i),j)+u(1,j);    
    end
    u(1,j) = u(1,j)/numel(classOne);
end
for j = 1:line
    for i = 1:numel(classOne)
        xita(1,j) = (Xtrain(classOne(i),j)-u(1,j))^2+xita(1,j);
    end
    xita(1,j) = xita(1,j)/numel(classOne);
end

%repeat the former steps for the classTwo
for j = 1:line
    for i = 1:numel(classTwo)
        u(2,j) = Xtrain(classTwo(i),j)+u(2,j);
    end
    u(2,j) = u(2,j)/numel(classTwo);
end
for j = 1:line
    for i = 1:numel(classTwo)
        xita(2,j) = (Xtrain(classTwo(i),j)-u(2,j))^2+xita(2,j);
    end
    xita(2,j) = xita(2,j)/numel(classTwo);
end

%This part is to use the ML estimation to do the prediction 
%P1 is the log form possibility for predict the sample as not spam
%P2 is the log form possibility for predic the sample as spam
%And calculte the errorrate for the training and testing set
%errorRateTest is the error rate for the test set
%errorRateTrain is the error rate for the training set

[Row, Line] = size(Xtest);
result = zeros(Row,1);
for i = 1:Row
    P1 = 0;
    P2 = 0;
    for j = 1:line
        P1 = P1+log(normpdf(Xtest(i,j),u(1,j),xita(1,j)^0.5));
    end  
    for j = 1:line
        P2 = P2+log(normpdf(Xtest(i,j),u(2,j),xita(2,j)^0.5));
    end
    P1 = P1 + log(1-lamML);
    P2 = P2 + log(lamML);
    if P1 >= P2
        result(i) = 0;
    else
        result(i) = 1;
    end
        
end
%calculate the error rate
error = 0;
for i = 1:Row
    if result(i) ~= ytest(i)
        error = error + 1;
    end
end
errorRateTest = error/Row

%repeat the former steps for the train set
[Row, Line] = size(Xtrain);
result = zeros(Row,1);
for i = 1:Row
    P1 = 0;
    P2 = 0;
    for j = 1:line
        P1 = P1+log(normpdf(Xtrain(i,j),u(1,j),xita(1,j)^0.5));
    end
    for j = 1:line
        P2 = P2+log(normpdf(Xtrain(i,j),u(2,j),xita(2,j)^0.5));
    end
    P1 = P1 + log(1-lamML);
    P2 = P2 + log(lamML);
    if P1 >= P2
        result(i) = 0;
    else
        result(i) = 1;
    end 
end
error = 0;
for i = 1:Row
    if result(i) ~= ytrain(i)
        error = error + 1;
    end
end
errorRateTrain = error/Row