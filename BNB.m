clear all; close all;
%load the data
load spamData;

%gennerate parameter alpha for Beta distribution
a = 0:0.5:100;
%estimation of the label prior using ML
lamML = sum(ytrain(:)==1)/numel(ytrain);    

Xbinary = zeros(size(Xtrain));  % initialize
%%%% This psrt is for binariztion of the feature%%%
for i = 1:numel(Xtest)
    if Xtest(i) > 0 
        Xtest(i) = 1;
    else
        Xtest(i) = 0;
    end
end
for i = 1:numel(Xtrain)
    if Xtrain(i) > 0 
        Xbinary(i) = 1;
    else
        Xbinary(i) = 0;
    end
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
%%%%% This part is to calculate the non-zero numbers of each feature in the two classes 
% N(1,j) saves the non-zeros number for classOne's jth feature
% N(2,j) saves the non-zeros number for classTwo's jth feature
[row, line] = size(Xbinary);
N = zeros(2,line);
for j = 1:line
    for i = 1:numel(classOne)   
        if Xbinary(classOne(i),j) == 1
            N(1,j) = N(1,j) + 1;
        end
    end
end
for j = 1:line
    for i = 1:numel(classTwo)   
        if Xbinary(classTwo(i),j) == 1
            N(2,j) = N(2,j) + 1;
        end
    end
end

%This part is to use the posterior predictive distribution to do the prediction 
%P1 is the log form possibility for predict the sample as not spam
%P2 is the log form possibility for predic the sample as spam
%and calculte the errorrate for the training and testing set
%errorRateTest is the error rate for the test set
%errorRateTrain is the error rate for the training set
for k = 1:length(a)
    P = zeros(2,line);
    for j = 1:line
        P(1,j) = (N(1,j)+a(k))/(length(classOne)+a(k)+a(k));
    end

    for j = 1:line
        P(2,j) = (N(2,j)+a(k))/(length(classTwo)+a(k)+a(k));
    end 
    [Row,Line] = size(Xbinary);
    result = zeros(Row,1);
    for i = 1:Row
        P1 = 0;
        P2 = 0;
         for j = 1:Line
            if Xbinary(i,j) == 1
                P1 = P1+log(P(1,j));
            else
                P1 = P1+log(1-P(1,j));
            end
        end 
        for j = 1:Line
            if Xbinary(i,j) == 1
                P2 = P2+log(P(2,j));
            else
                P2 = P2+log(1-P(2,j));
            end
        end
%add the label prior 
        P1 = P1+log(1-lamML);
        P2 = P2+log(lamML);
        if P1 >= P2
%rusult saves the prediction for each samples
            result(i) = 0;
        else
            result(i) = 1;
        end
    end
%calculate the error rate
    error = 0;
    for i = 1:Row
        if result(i) ~= ytrain(i)
            error = error + 1;
        end
    end
    errorRateTrain(k) = error/Row;
    
%%% repeat the former step for the test set
    [Row,Line] = size(Xtest);
    result = zeros(Row,1);
    for i = 1:Row
        P1 = 0;
        P2 = 0;
        for j = 1:Line
            if Xtest(i,j) == 1
                P1 = P1+log(P(1,j));
            else
                P1 = P1+log(1-P(1,j));
            end
        end
        for j = 1:Line
            if Xtest(i,j) == 1
                P2 = P2+log(P(2,j));
            else
                P2 = P2+log(1-P(2,j));
            end
        end
        P1 = P1+log(1-lamML);
        P2 = P2+log(lamML);
        if P1 >= P2
            result(i) = 0;
        else
            result(i) = 1;
        end
    end
    error = 0;
    for i = 1:Row
        if result(i) ~= ytest(i)
            error = error + 1;
        end
    end
    errorRateTest(k) = error/Row;
end

TestR = [errorRateTest(3),errorRateTest(21),errorRateTest(201)];
TrainR = [errorRateTrain(3),errorRateTrain(21),errorRateTrain(201)];

plot(a,errorRateTrain)
hold on
plot(a,errorRateTest)
title('errorRate')
legend('errorRateTrain','errorRateTest')