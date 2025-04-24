clc;
close all;
clear;
currentFolder = pwd;
addpath(genpath(currentFolder));
resultdir = 'D:\datasetandcode\incomplete multiview datasets\';
datadir='D:\datasetandcode\incomplete multiview datasets\';
% resultdir = 'C:\Users\wangs\Desktop\Incomplete Multi-view Clustering code\Incomplete\';
% datadir='C:\Users\wangs\Desktop\Incomplete Multi-view Clustering code\Incomplete\';
% dataname={'MSRCV1', 'ORL', '20newsgroups','COIL20-3v', 'handwritten', ...
%     '100leaves', 'yale_mtv_2', 'Wikipedia'};
% dataname={'MSRCV1_3v', 'handwritten_3v', 'Wiki', 'scene-15'};
%%dataname={'AwA_fea', 'Caltech101-7','Caltech101-20','Caltech101-all_fea','CCV','Mfeat','MNIST_fea','MNIST_fea_sort','NUSWIDEOBJ','ORL_mtv','SUNRGBD_fea','Wiki_fea','YoutubeFace_sel_fea'};
%%dataname={'100Leaves','NGs','prokaryotic','proteinFold','synthetic3d','uci-digit','WebKB'};
% dataname = {'ALOI-100','BBC','BBCSport','handwritten','Handwritten_numerals','Hdigit','uci-digit'};
dataname = {'Caltech101-all','CCV','ALOI3v','MNIST-10k','Cornell','bbcsport4vbigRnSp','Yale','3sources','Cornell','Washington','WikipediaArticles','BBC4view_685','bbcsport12RnSp','NH_p4660','NUS','100leaves','Coil20','Animal','Hdigit','NUSWIDE','SUNRGBD','ORL_mtv','WebKB','BBCSport','handwritten','Handwritten_numerals'};
n_dataset = length(dataname); % number of the datasets
% for idata = 1:n_dataset-(n_dataset-1)
for idata = 1:1
    % read dataset
    dataset_file = [datadir, cell2mat(dataname(idata)),'.mat'];
    load(dataset_file);
    %X = data;
    %Y = truelabel{1};
%      X{1}=double(X1);
%      X{2}=double(X2);
%      X{3}=double(X3);
    V = length(X); % the number of views
    oriData = cell(V,1);
    oriTruelabel = cell(V,1);
    for v = 1:V
        oriData{v} = X{v}';%X{v}/X{v}'
        oriTruelabel{v} = Y;%gt/Y
    end
    clear X gt;
    N = size(oriData{1},2); % the number of instances
    n_view = length(oriData);
    perGrid = [1:-0.1:0]; %  the percentage of paired instances 
    misingExampleVector = randperm(N);
    MissingStatus = zeros(N, n_view); % indicate the missing status of instance in each view
    for id = 1:N
        missingViewVector = randi([0, 1], n_view, 1, 'int8');%'int8'
        while(0 == sum(missingViewVector) || n_view == sum(missingViewVector))
            % in case of all views mising
            missingViewVector = randi([0,1], n_view,1, 'int8');
        end
        MissingStatus(id, :) = missingViewVector;
    end
    for per_iter = 2:10%length(perGrid)
        per = perGrid(per_iter);% partial example ratio
        miss_n = fix((1-per)*N); % the number of missing instances
%             if miss_n~=1*N && miss_n~=0.9*N && miss_n~=0.8*N && miss_n~=0.7*N && miss_n~=0.6*N && miss_n~=0.5*N && miss_n~=0.4*N && miss_n~=0.3*N && miss_n~=0.2*N && miss_n~=0.1*N && miss_n~=0*N
%                 miss_n=miss_n+1;
%             end
        perMisingExampleVector = misingExampleVector(1: miss_n);
        data = oriData;
        full_index = cell(1,n_view);
        for v = 1:n_view
            full_index{v} = zeros(N,1);
        end
        for id = 1: miss_n
            for j = 1 : n_view
                if 0 == MissingStatus(misingExampleVector(id), j)
                    data{j}(:,misingExampleVector(id)) = nan;
                    full_index{j}(misingExampleVector(id)) = 1;%缺失的列标记为1
                end
            end
        end
        index = cell(1, n_view);
        for v = 1:n_view
            index{v} = find(full_index{v}==0);
        end
        truelabel = oriTruelabel;
        save([resultdir,char(dataname(idata)),'_Per',num2str(per),'.mat'],'data','truelabel','index','MissingStatus');
        clear data truelabel index;
    end
end
