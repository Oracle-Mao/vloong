from pyod.models.auto_encoder_torch import AutoEncoder, check_array, inner_autoencoder, check_is_fitted, \
    pairwise_distances_no_broadcast

class Car_AutoEncoder(AutoEncoder):
    
    '''
    使用autoencoder 来进行模型的训练，默认采用无监督的训练方式
    '''
    
    def fit(self, X, y=None):
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        n_samples, n_features = X.shape[0], X.shape[1] #获取样本个数和特征个数

        # 是否进行预处理操作
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            train_set = PyODDataset(X=X, mean=self.mean, std=self.std)

        else:
            train_set = PyODDataset(X=X)
        #构建数据生成器
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=False)

        # initialize the model ,初始化模型
        #hidden_neurons列表，可选（默认值为[64， 32]）每个隐藏层的神经元数。因此，该网络的结构为[n_features，64，32，32，64，n_features]
        #hidden_activationstr，可选（默认值='relu'）用于隐藏层的激活函数。所有隐藏层都强制使用相同类型的激活
        #batch_norm布尔值，可选（默认值为 True）是否应用批量规范化。
        #dropout_rate浮点数 （0.， 1），可选（默认值 = 0.2）要跨所有层使用的分级。
        
        self.model = inner_autoencoder(
            n_features=n_features,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            hidden_activation=self.hidden_activation)

        #将model放入device中
        self.model = self.model.to(self.device)

        # 训练自动编码器以找到最佳编码器
        self._train_autoencoder(train_loader)

        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X)#获得输入样本的异常得分
        
        self._process_decision_scores()  
        return self

    def decision_function(self, X): 
        """使用拟合的检测器预测X的原始异常分数。

            输入样本的异常分数是基于不同的检测器算法。为保持一致性，离群值分配为异常分数越大的。
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            形状的numpy数组（n_samples，n_features）训练输入样本。仅接受稀疏矩阵，如果它们由基础估计器支持。

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            形状的numpy数组（n_samples，）输入样本的异常得分。
        """
        #对估算器执行is_fitted验证。通过验证是否存在拟合属性（以下划线结尾）来检查估计量是否拟合，否则通过给定消息引发NotFittedError。此实用程序旨在由估计器本身在内部使用，通常在其自己的预测/变换方法中使用。
        check_is_fitted(self, ['model', 'best_model_dict'])
        # X = check_array(X)

        # note the shuffle may be true but should be False
        if self.preprocessing:
            dataset = PyODDataset(X=X, mean=self.mean, std=self.std)
        else:
            dataset = PyODDataset(X=X)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False) #要设置为False
        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([X.shape[0], ])#形状为（X.shape[0],)
        with torch.no_grad():
            for data, data_idx in dataloader:
                data_cuda = data.to(self.device).float()
                # this is the outlier score
                outlier_scores[data_idx] = pairwise_distances_no_broadcast(
                    data, self.model(data_cuda).cpu().numpy())

        return outlier_scores