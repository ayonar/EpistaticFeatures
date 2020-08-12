% Finding Higher order intereactions between features after training artificial neural network for regression
% 02/05/2018 by A. Yonar

% Idea: Using compressed sensing for finding higher order sparse feature interactions.

%% Simulate Forward Problem & Generate Dataset

% y = w*X + 0.5*(X'JX) where w is weight vector J is intereaction weight
% matrix

P = 30; % number of features
N = 10000; % number of samples
w = linspace(0,1,P); % linearly increasing impoertance features. We can do exponential.

% generate sparse 2-way intereactions
J = zeros(P,P);
s=5;
mysign = randi([0,1],1,s); mysign(mysign==0)=-1;
entries = randi([1,30],2,s);
% make sure diagonals are zero -- check that.
for k=1:s
    J(entries(1,k),entries(2,k)) = mysign(k);
    J(entries(2,k),entries(1,k)) = mysign(k);
end

%% Generate Dataset with above weights
X = zeros(P,1);
a=1;
b=2;
Inputs=zeros(N,P);
Targets=zeros(N,1);
for i = 1:N
    %X = a + (b-a).*rand(P,1);
    X = randi([1,10],30,1);
    y = w*X + 0.5*X'*J*X;
    Inputs(i,:)=X;
    Targets(i)=y;
end

clear y
%% Split train data and test data
x_train = Inputs(1:0.9*N,:);
y_train = Targets(1:0.9*N);
x_test = Inputs(0.9*N+1:end,:);
y_test = Targets(0.9*N+1:end);

%% 

net = feedforwardnet(20);
[net,tr] = train(net,x_train',y_train');

%% Network & Traning Details
% Construct Network Layers
layers = [
    imageInputLayer([30 1])
    
% %     convolution2dLayer(3,16,'Padding',1)
% %     batchNormalizationLayer
% %     reluLayer   
% %     
% %     maxPooling2dLayer(2,'Stride',2)
% %     
% %     convolution2dLayer(3,32,'Padding',1)
% %     batchNormalizationLayer
% %     reluLayer   
    
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
% %     fullyConnectedLayer(10)
% %     reluLayer
    fullyConnectedLayer(1)
    softmaxLayer
    classificationLayer];

% Traninig Options
% % % options = trainingOptions('sgdm','Verbose',false, ...
% % %     'MaxEpochs',3, ...
% % %     'InitialLearnRate',0.03);

% % % options = trainingOptions('sgdm',...
% % %     'MaxEpochs',3, ...
% % %     'ValidationData',valDigitData,...
% % %     'ValidationFrequency',30,...
% % %     'Verbose',false,...
% % %     'Plots','training-progress');

miniBatchSize=128;
options = trainingOptions('sgdm',...
    'InitialLearnRate',0.01,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',50,...
    'LearnRateDropFactor',0.5,...
    'MaxEpochs',400,...
    'MiniBatchSize',miniBatchSize,...
    'ValidationFrequency',30,...
    'ValidationData',valDigitData,...
    'ValidationPatience',Inf,...
    'Verbose',false,...
    'Plots','training-progress');

