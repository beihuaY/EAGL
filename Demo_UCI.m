clear,clc;
addpath('ClusteringMeasure','Datasets','Tools');
resultdir1 = 'Results/';
if (~exist('Results', 'file'))
    mkdir('Results');
    addpath(genpath('Results/'));
end
resultdir2 = 'aResults/';
if (~exist('aResults', 'file'))
    mkdir('aResults');
    addpath(genpath('aResults/'));
end
%fprintf('Demo of FEMSRL on my_UCI\n')
%load('Datasets\my_UCI.mat');
% dataname={'Caltech101-all','WebKB','Caltech101-7','bbcsport4vbigRnSp','Cornell','BDGP_fea','NH_p4660','Animal','Yale','3sources','Cornell','BBC4view_685','Washington','WikipediaArticles','bbcsport12RnSp','NUS','100leaves','Coil20','BDGP_fea','ORL_mtv'};
dataname={'bbcsport4vbigRnSp'};
% numname = {'_Per0.1', '_Per0.2', '_Per0.3', '_Per0.4','_Per0.5', '_Per0.6', '_Per0.7', '_Per0.8', '_Per0.9'};
 numname = { '_Per0.9'};
datadir='./Datasets/';
num_name=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
ResBest = zeros(9, 8);
ResStd = zeros(9, 8);
for dataIndex =1:1
datafile = [datadir, cell2mat(dataname(1)), cell2mat(numname(dataIndex)), '.mat'];
load(datafile);
num_name1 = num_name(dataIndex);
gt = truelabel{1};
[X,ind] = findindex(data, index);

v = length(X);
for i = 1:v
    X{i} = X{i}./(repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1)+1e-8);
end
k = length(unique(gt));

%A = cell(1,v); 
lambda = [0.1]; %α
lambdaa = [0.001 0.01 0.1 1 10 100 1000]; %[0.001, 0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 50];λ
numanchor= [k 2*k 3*k 4*k];
for anchor =2:2
numanchor1= numanchor(anchor);%[k 2k 3k]
for LambdaIndex1 = 1:1
lambda1 = lambda(LambdaIndex1);
   for LambdaIndex2 = 1:7
       lambda2 = lambdaa(LambdaIndex2);

rng(4396,'twister');

fprintf('params: paired ratio=%f \t numanchor=%d \t alpha=%f \t lambda=%f\n',num_name1, numanchor1, lambda1,lambda2);

tic;
[U_results,V_results,Z_results,Q,obj]  = FEMSRL(X,numanchor1,gt,lambda1,lambda2,ind);
t=toc;
rng(1234,'twister') % set random seed for re-production
 for rep = 1 : 5
     labels=litekmeans(Q, k, 'MaxIter', 100,'Replicates',10);
     res(rep, : ) = Clustering8Measure(gt, labels);%res:ACC nmi Purity Fscore Precision Recall AR Entropy
 end
 tempResBest(dataIndex, : ) = mean(res);%9行8列，行：每个缺失率
 tempResStd(dataIndex, : ) = std(res);
 ACC(LambdaIndex1, LambdaIndex2) = tempResBest(dataIndex, 1);
 NMI(LambdaIndex1, LambdaIndex2) = tempResBest(dataIndex, 2);
 Purity(LambdaIndex1, LambdaIndex2) = tempResBest(dataIndex, 3);
 Fscore(LambdaIndex1, LambdaIndex2) = tempResBest(dataIndex, 4);
 fprintf('result:\tACC:%f, NMI:%f, Purity:%f, Fscore:%f, times:%f\n',ACC(LambdaIndex1, LambdaIndex2),NMI(LambdaIndex1, LambdaIndex2),Purity(LambdaIndex1, LambdaIndex2),Fscore(LambdaIndex1, LambdaIndex2),t);
 save([resultdir1, char(dataname(1)), char(numname(dataIndex)), '-l1=', num2str(lambda1), '-l2=', num2str(lambda2), ...
                    '-acc=', num2str(tempResBest(dataIndex,1)), '_result.mat'], 'tempResBest', 'tempResStd');
                for tempIndex = 1 : 8
                    if tempResBest(dataIndex, tempIndex) > ResBest(dataIndex, tempIndex)
                        ResBest(dataIndex, tempIndex) = tempResBest(dataIndex, tempIndex);
                        ResStd(dataIndex, tempIndex) = tempResStd(dataIndex, tempIndex);
                    end
                end

   end
end
        PResBest = ResBest(dataIndex, :);
        PResStd = ResStd(dataIndex, :);
        save([resultdir2, char(dataname(1)), char(numname(dataIndex)), 'ACC_', num2str(max(ACC(:))),'k_',num2str(numanchor1), '_result.mat'], 'ACC', 'NMI', 'Purity', 'Fscore', ...
             'PResBest', 'PResStd','ResBest','ResStd');
end
end