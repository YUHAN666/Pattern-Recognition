clear all;
load spamData;
%%%%This part is to log transform the feature%%%
for i = 1:numel(Xtrain)
    Xtrain(i) = log(Xtrain(i)+0.1);
end
for i = 1:numel(Xtest)
    Xtest(i) = log(Xtest(i)+0.1);
end
%%%%%This part is to generate the K%%%%%%%%
K1 = [1:9];
K2 = [10:5:100];
K = [K1,K2];

%%%%%%This part is to use KNN to make the prediction%%%

[row_train,line_train] = size(Xtrain);
[row_test, line_test] = size(Xtest);
result = zeros(row_test,length(K));
distance = zeros(row_train,2);
for k = 1:length(K)
    for j = 1:row_test
        for i = 1:row_train
           %calculate the distance between point j and all the other points
           distance(i,1) = norm(Xtrain(i,:)-Xtest(j,:),2);
           %save the label for the point
           distance(i,2)  = ytrain(i); 
        end
        %order the distance from small to large
        disord = sortrows(distance,1);
        %pick the k points with cloest distance
        dismin = disord(1:K(k),2);
        %count the number of non-zero label in the k samples
        count = sum(dismin);
        %make the prediction according to the count
        if count > K(k)/2
            result(j,k) = 1;
        else
            result(j,k) = 0;
        end
    end
end
%%%%calculate the error rate
errorRateTest = zeros(length(K),1);
for k = 1:length(K)
    errorRateTest(k,1) = sum(abs(ytest(:,1)-result(:,k)))/row_test;
end

%%%%%%repeat the former step for the training set%%%%%%%%%%
result = zeros(row_train,length(K));
distance = zeros(row_train,2);
for k = 1:length(K)
    for j = 1:row_train
        for i = 1:row_train
           dis = Xtrain(i,:)-Xtrain(j,:);
           distance(i,1) = norm(dis,2);
           distance(i,2)  = ytrain(i); 
        end
        disord = sortrows(distance,1);
        dismin = disord(1:K(k),2);
        count = sum(dismin);
        if count > K(k)/2
            result(j,k) = 1;
        else
            result(j,k) = 0;
        end
    end
end
errorRateTrain = zeros(length(K),1);
for k = 1:length(K)
    errorRateTrain(k,1) = sum(abs(ytrain(:,1)-result(:,k)))/row_train;
end
plot(K,errorRateTrain)
hold on
plot(K,errorRateTest)
legend('errorRateTrain','errorRateTest')
TestR = [errorRateTest(1),errorRateTest(10),errorRateTest(28)];
TrainR = [errorRateTrain(1),errorRateTrain(10),errorRateTrain(28)];

