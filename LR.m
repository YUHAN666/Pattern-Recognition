clear all;
load spamData;
%%%This part is to log transform the features%%
for i = 1:numel(Xtrain)
    Xtrain(i) = log(Xtrain(i)+0.1);
end
for i = 1:numel(Xtest)
    Xtest(i) = log(Xtest(i)+0.1);
end
%%%%This part is to generate the lambda%%%
lamda1 = [1:9];
lamda2 = [10:5:100];
lamda = [lamda1,lamda2];
leng = length(lamda);

%%%This part is to initialize the matrixs%%%%%%
[N, line] = size(Xtrain);
one = ones(N,1);
X = [one,Xtrain];       % add a feature 1 so that w1 is the bias
Y = ytrain;
u = zeros(N,1);
w = zeros(line+1,leng);
S = zeros(N,N);

%%This part is to update the w using training set
%each colomn saves the w for each lambda
for j = 1:leng
    for k = 1:100        %iteration 50 times for each lamda
        for i = 1:N
            u(i,1) = 1/(1+exp(-(X(i,:)*w(:,j))));
            S(i,i) =  u(i,1)*(1-u(i,1));
        end
        v = ones(1,line+1);
        v(1,1) = 0;
        H = X'*S*X+lamda(j)*diag(v);    % Hessian Matrix 
        g = X'*(u-Y)+lamda(j)*[0;w(2:line+1,j)];    % Grandient Matrix
        w(:,j) = w(:,j) - H^(-1)*g;     %iteration for w
    end
end

%%%%%This part is to calculate the error rate for test set %%%%%%
%%each colomn of the errorRateTest saves error rate for each lambda
[N, line] = size(Xtest);
one = ones(N,1);
X = [one,Xtest];
Y = ytest;
for j = 1:leng
    count = 0;
    for i = 1:N
        u = 1/(1+exp(-(X(i,:)*w(:,j))));
        if u >= 0.5
            y = 1;
        else
            y = 0;
        end
        if y ~= Y(i,1)
            count = count+1;
        end
    end
    errorRateTest(j) = count/N; 
end

%%%%%%%%This part is to calculate the error rate for the training set%%%%%%
%%each colomn of the errorRateTrain saves error rate for each lambda
[N, line] = size(Xtrain);
one = ones(N,1);
X = [one,Xtrain];
Y = ytrain;

for j = 1:leng
    count = 0;
    for i = 1:N
        u = 1/(1+exp(-(X(i,:)*w(:,j))));
        if u >= 0.5
            y = 1;
        else
            y = 0;
        end
        if y ~= Y(i,1)
            count = count+1;
        end
    end
    errorRateTrain(j) = count/N; %error每一列存储一个lamda的error rate
end

plot(lamda, errorRateTrain)
hold on
plot(lamda, errorRateTest)
title('errorRate')
legend('errorRateTrain','errorRateTest')
TestR = [errorRateTest(1),errorRateTest(10),errorRateTest(28)];
TrainR = [errorRateTrain(1),errorRateTrain(10),errorRateTrain(28)];

