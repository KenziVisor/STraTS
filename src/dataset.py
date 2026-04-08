import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from utils import CycleIndex
import os
import re

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ACCEPTED_ID_COLUMNS = ['ts_id', 'icustay_id', 'ICUSTAY_ID']


def canonicalize_stay_id_series(series):
    def normalize_value(value):
        if pd.isna(value):
            return value
        value = str(value).strip()
        if re.fullmatch(r'[+-]?\d+\.0+', value):
            value = value.split('.', 1)[0]
        if value.startswith('+'):
            value = value[1:]
        return value

    return series.apply(normalize_value)


def normalize_id_column(df, df_name):
    present_id_columns = [col for col in ACCEPTED_ID_COLUMNS if col in df.columns]
    if not present_id_columns:
        raise KeyError(
            f"{df_name} is missing a stay identifier column. "
            f"Accepted columns: {ACCEPTED_ID_COLUMNS}. "
            f"Present columns: {list(df.columns)}"
        )

    normalized_df = df.copy()
    normalized_ids = pd.DataFrame(
        {col: canonicalize_stay_id_series(normalized_df[col]) for col in present_id_columns},
        index=normalized_df.index,
    )
    canonical_ts_id = normalized_ids.bfill(axis=1).iloc[:, 0]

    for col in present_id_columns:
        mismatch = canonical_ts_id.notna() & normalized_ids[col].notna() & (
            canonical_ts_id != normalized_ids[col]
        )
        if mismatch.any():
            sample_rows = normalized_ids.loc[mismatch, present_id_columns].head(5)
            raise ValueError(
                f"{df_name} has inconsistent stay identifier columns among {present_id_columns}. "
                f"Sample rows: {sample_rows.to_dict(orient='records')}"
            )

    normalized_df['ts_id'] = canonical_ts_id
    drop_columns = [col for col in present_id_columns if col != 'ts_id']
    if drop_columns:
        normalized_df = normalized_df.drop(columns=drop_columns)
    return normalized_df


def _canonicalize_id_array(values, array_name):
    normalized_values = canonicalize_stay_id_series(pd.Series(values, copy=False))
    if normalized_values.isna().any():
        raise ValueError(f"{array_name} contains missing stay identifiers after normalization.")
    return normalized_values.to_numpy(dtype=object)


def _stay_id_sort_key(value):
    if pd.isna(value):
        return (2, 0, '')
    if re.fullmatch(r'[+-]?\d+', value):
        return (0, int(value), '')
    return (1, 0, value)


def _sorted_unique_ids(values):
    unique_values = pd.Index(values).unique().tolist()
    return np.array(sorted(unique_values, key=_stay_id_sort_key), dtype=object)


def _intersect_canonical_ids(left_ids, right_ids):
    right_id_set = set(right_ids)
    return _sorted_unique_ids([stay_id for stay_id in left_ids if stay_id in right_id_set])


def _setdiff_canonical_ids(left_ids, right_ids):
    right_id_set = set(right_ids)
    return _sorted_unique_ids([stay_id for stay_id in left_ids if stay_id not in right_id_set])


def _format_id_sample(values, max_items=10):
    sample = list(values)[:max_items]
    if len(values) > max_items:
        return sample + ['...']
    return sample


class Dataset:
    def __init__(self, args) -> None:
        # read data
        filepath = os.path.normpath(os.path.join(
            SRC_DIR, '..', 'data', 'processed', args.dataset + '.pkl'
        ))
        data, oc, train_ids, val_ids, test_ids = pickle.load(open(filepath,'rb'))
        data = normalize_id_column(data, 'processed events')
        oc = normalize_id_column(oc, 'processed outcomes')
        train_ids = _canonicalize_id_array(train_ids, 'train_ids')
        val_ids = _canonicalize_id_array(val_ids, 'val_ids')
        test_ids = _canonicalize_id_array(test_ids, 'test_ids')
        run, totalruns = list(map(int, args.run.split('o')))
        num_train = int(np.ceil(args.train_frac*len(train_ids)))
        start = int(np.linspace(0,len(train_ids)-num_train,totalruns)[run-1])
        train_ids = train_ids[start:start+num_train]
        num_val = int(np.ceil(args.train_frac*len(val_ids)))
        start = int(np.linspace(0,len(val_ids)-num_val,totalruns)[run-1])
        val_ids = val_ids[start:start+num_val]
        args.logger.write('\nPreparing dataset '+args.dataset)
        static_varis = self.get_static_varis(args.dataset)
        if args.dataset=='mimic_iii':
            # Filter labeled data in first 24h and fill missing age for old patients.
            data = data.loc[(data.minute>=0)&(data.minute<=24*60)]
            data.loc[(data.variable=='Age')&(data.value>200), 'value'] = 91.4

        # keep variables seen in training set only
        train_variables = data.loc[data.ts_id.isin(train_ids)].variable.unique()
        all_variables = data.variable.unique()
        delete_variables = np.setdiff1d(all_variables, train_variables)
        args.logger.write('Removing variables not in training set: '+str(delete_variables))
        data = data.loc[data.variable.isin(train_variables)]
        curr_ids = data.ts_id.unique()
        train_ids = _intersect_canonical_ids(train_ids, curr_ids)
        val_ids = _intersect_canonical_ids(val_ids, curr_ids)
        test_ids = _intersect_canonical_ids(test_ids, curr_ids)
        args.logger.write('# train, val, test TS: '+str([len(train_ids), len(val_ids), len(test_ids)]))
        sup_ts_ids = np.concatenate((train_ids, val_ids, test_ids))
        raw_ts_id_to_ind = {ts_id:i for i,ts_id in enumerate(sup_ts_ids)}
        data = data.loc[data.ts_id.isin(sup_ts_ids)]
        data['ts_ind'] = data['ts_id'].map(raw_ts_id_to_ind)

        # Get y and N
        latent_df = pd.read_csv(args.latent_csv_path)
        latent_df = normalize_id_column(latent_df, 'latent CSV')

        sup_ts_ids_str = list(sup_ts_ids)
        str_ts_id_to_ind = {ts_id: i for i, ts_id in enumerate(sup_ts_ids_str)}

        latent_df = latent_df.loc[latent_df['ts_id'].isin(sup_ts_ids_str)].copy()

        missing_ids = set(sup_ts_ids_str) - set(latent_df['ts_id'])
        if missing_ids:
            missing_id_sample = _format_id_sample(sorted(missing_ids))
            raise ValueError(
                f"latent CSV is missing labels for {len(missing_ids)} supervised ts_id values. "
                f"Sample missing IDs: {missing_id_sample}"
            )

        if latent_df['ts_id'].duplicated().any():
            dup_ids = latent_df.loc[latent_df['ts_id'].duplicated(keep=False), 'ts_id']
            dup_id_sample = _format_id_sample(dup_ids.drop_duplicates().tolist())
            raise ValueError(
                f"latent CSV has duplicate labels for {dup_ids.nunique()} normalized ts_id values. "
                f"Sample duplicate IDs: {dup_id_sample}"
            )

        target_columns = [c for c in latent_df.columns if c != 'ts_id']
        latent_df['ts_ind'] = latent_df['ts_id'].map(str_ts_id_to_ind)
        latent_df = latent_df.sort_values(by='ts_ind')
        y = latent_df[target_columns].to_numpy(dtype=np.float32)

        args.num_targets = len(target_columns)
        args.target_columns = target_columns
        N = len(sup_ts_ids_str)

        # To save
        self.N = N
        self.y = y
        self.args = args
        self.static_varis = static_varis
        self.splits = {'train':[raw_ts_id_to_ind[i] for i in train_ids],
                       'val':[raw_ts_id_to_ind[i] for i in val_ids],
                       'test':[raw_ts_id_to_ind[i] for i in test_ids]}
        self.splits['eval_train'] = self.splits['train'][:2000]
        self.sup_ts_ids = sup_ts_ids_str
        self.target_columns = target_columns
        self.train_cycler = CycleIndex(self.splits['train'], args.train_batch_size)

        train_y = y[self.splits['train']]
        val_y = y[self.splits['val']]
        test_y = y[self.splits['test']]

        num_train = train_y.shape[0]
        train_pos = train_y.sum(axis=0)
        val_pos = val_y.sum(axis=0)
        test_pos = test_y.sum(axis=0)

        safe_den = np.where(train_pos > 0, train_pos, 1.0)
        pos_class_weight = (num_train - train_pos) / safe_den
        pos_class_weight = np.where(train_pos > 0, pos_class_weight, 0.0)

        args.pos_class_weight = pos_class_weight.astype(np.float32)

        args.logger.write('pos class weight per target: ' + str(dict(zip(target_columns, args.pos_class_weight))))
        args.logger.write('positive rate per target in train: ' + str(
            dict(zip(target_columns, (train_pos / max(len(train_y), 1)).tolist()))))
        args.logger.write('positive rate per target in val: ' + str(
            dict(zip(target_columns, (val_pos / max(len(val_y), 1)).tolist()))))
        args.logger.write('positive rate per target in test: ' + str(
            dict(zip(target_columns, (test_pos / max(len(test_y), 1)).tolist()))))

        if 'llm' in args.model_type:
            self.data = data
            return

        # Get static data with missingness indicator.
        data = self.get_static_data(data)

        # Trim to max len.
        if args.model_type in ['strats', 'istrats']:
            data = data.sample(frac=1)
            data = data.groupby('ts_id').head(args.max_obs)
        elif args.model_type in ['grud', 'interpnet']:
            timestamps = data[['ts_id','minute']].drop_duplicates().sample(frac=1)
            timestamps = timestamps.groupby('ts_id').head(args.max_timesteps)
            data = data.merge(timestamps, on=['ts_id','minute'], how='inner')

        # normalize if not aggregating, also get max_minute for strats
        args.finetune = args.load_ckpt_path is not None
        if args.finetune:
            pt_var_path = os.path.join(os.path.dirname(args.load_ckpt_path),
                                       'pt_saved_variables.pkl')
            variables, means_stds, max_minute = pickle.load(open(pt_var_path,'rb'))
        if args.model_type in ['strats','istrats','grud','interpnet']:
            if not(args.finetune):
                means_stds = data.loc[data.ts_id.isin(train_ids)].groupby(
                                    'variable').agg({'value':['mean', 'std']})
                means_stds.columns = [col[1] for col in means_stds.columns]
                means_stds.loc[means_stds['std']==0, 'std'] = 1
                max_minute = data['minute'].max()
            data = data.merge(means_stds.reset_index(), on='variable', how='left')
            data['value'] = (data['value']-data['mean'])/data['std']

        # prepare time series inputs
        if not(args.finetune):
            variables = data.variable.unique()
        var_to_ind = {v:i for i,v in enumerate(variables)}
        V = len(variables)
        args.V = V
        args.logger.write('# TS variables: '+str(V))
        if args.model_type in ['gru', 'tcn', 'sand']:
            # get hourly agg ts with missingness and time since last obs
            data['minute'] = data['minute'].apply(lambda x:max(1, int(np.ceil(x/60)))-1)
            T = data.minute.max()+1
            args.T = T
            args.logger.write('# intervals: '+str(T))
            values = np.zeros((N,T,V))
            obs = np.zeros((N,T,V))
            for row in tqdm(data.itertuples()):
                vind = var_to_ind[row.variable]
                tstep = row.minute
                values[row.ts_ind, tstep, vind] = row.value
                obs[row.ts_ind, tstep, vind] = 1
            # Generate delta.
            delta = np.zeros((N,T,V))
            delta[:,0,:] = obs[:,0,:]
            for t in range(1,T):
                delta[:,t,:] = obs[:,t,:]*0 + (1-obs[:,t,:])*(1+delta[:,t-1,:])
            delta = delta/T
            # mean fill obs
            train_ind = self.splits['train']
            means = (values[train_ind] * obs[train_ind]).sum(axis=(0, 1)) / \
                    obs[train_ind].sum(axis=(0, 1))
            values = values*obs + (1-obs)*means.reshape((1,1,V))
            # normalize values
            means = values[train_ind].mean(axis=(0,1), keepdims=True)
            stds = values[train_ind].std(axis=(0,1), keepdims=True)
            stds = (stds==0)*1 + (stds>0)*stds
            values = (values-means)/stds
            self.X = np.concatenate((values, obs, delta), axis=-1)
        elif args.model_type in ['strats', 'istrats']:
            values = [[] for i in range(N)]
            times = [[] for i in range(N)]
            varis = [[] for i in range(N)]
            data['minute'] = data['minute']/max_minute*2-1
            for row in data.itertuples():
                values[row.ts_ind].append(row.value)
                times[row.ts_ind].append(row.minute)
                varis[row.ts_ind].append(var_to_ind[row.variable])
            self.values, self.times, self.varis = values, times, varis
        elif args.model_type in ['grud','interpnet']:
            if args.model_type=='grud':
                deltas = [[] for i in range(N)]
            elif args.model_type=='interpnet':
                times = [[] for i in range(N)]
                holdout_masks = [[] for i in range(N)]
            values = [[] for i in range(N)]
            mask = [[] for i in range(N)]
            for ts_ind, curr_data in data.groupby('ts_ind'):
                curr_times = sorted(list(curr_data.minute.unique()))
                time2idx = {t:i for i,t in enumerate(curr_times)}
                T = len(curr_times)
                curr_values, curr_mask = np.zeros((T,V)),np.zeros((T,V))
                for row in curr_data.itertuples():
                    time_idx = time2idx[row.minute]
                    vind = var_to_ind[row.variable]
                    curr_values[time_idx, vind] = row.value
                    curr_mask[time_idx, vind] = 1
                if args.model_type=='grud':
                    curr_delta = np.zeros((T,V))
                    for t in range(1,T):
                        curr_delta[t,:] = curr_times[t] - curr_times[t-1] + \
                                        (1 - curr_mask[t-1]) * curr_delta[t-1,:]
                    deltas[ts_ind] = curr_delta/(24*60*60) # days
                elif args.model_type=='interpnet':
                    times[ts_ind] = list(np.array(curr_times)/60) # hours
                    curr_mask[0,:] = 1
                    hmask = np.copy(curr_mask)
                    for j in range(args.V):
                        obs_time_indices = np.argwhere(curr_mask[:,j]).reshape(-1)
                        num_to_mask = int(0.2*len(obs_time_indices))
                        if num_to_mask>0:
                            to_mask = np.random.choice(obs_time_indices, num_to_mask, replace=False)
                            hmask[to_mask,j] = 0
                    holdout_masks[ts_ind] = hmask
                values[ts_ind] = curr_values
                mask[ts_ind] = curr_mask
            self.values, self.mask = values, mask
            if args.model_type=='grud':
                self.deltas = deltas
            elif args.model_type=='interpnet':
                self.times = times
                self.holdout_masks = holdout_masks

    def get_supervised_ts_ids(self):
        return self.sup_ts_ids

    def get_static_varis(self, dataset):
        if dataset=='mimic_iii':
            static_varis = ['Age', 'Gender']
        elif dataset=='physionet_2012':
            static_varis = ['Age', 'Gender', 'Height', 'ICUType_1',
                            'ICUType_2', 'ICUType_3', 'ICUType_4']
        return static_varis

    def get_static_data(self, data):
                # Get static data with missingness indicator.
        static_ii = data.variable.isin(self.static_varis)
        static_data = data.loc[static_ii]
        data = data.loc[~static_ii] # remove static vars from data
        static_var_to_ind = {v:i for i,v in enumerate(self.static_varis)}
        D = len(static_var_to_ind)
        if self.args.dataset=='physionet_2012':
            D+=2
            self.static_varis += ['Gender_missing', 'Height_missing']
        demo = np.zeros((self.N, D))
        for row in tqdm(static_data.itertuples()):
            var_ind = static_var_to_ind[row.variable]
            demo[row.ts_ind, var_ind] = row.value
            if self.args.dataset=='physionet_2012':
                if row.variable=='Gender':
                    demo[row.ts_ind, D-2] = 1
                elif row.variable=='Height':
                    demo[row.ts_ind, D-1] = 1
        # mean fill missing static values
        if self.args.dataset=='physionet_2012':
            static_data_train = static_data.loc[static_data.ts_ind.isin(self.splits['train'])]
            gender_mean = static_data_train.loc[static_data_train.variable=='Gender']['value'].mean()
            height_mean = static_data_train.loc[static_data_train.variable=='Height']['value'].mean()
            del static_data_train
            gender_mask = (1-demo[:,D-2]).astype(bool)
            demo[gender_mask, static_var_to_ind['Gender']] = gender_mean
            height_mask = (1-demo[:,D-1]).astype(bool)
            demo[height_mask, static_var_to_ind['Height']] = height_mean
        # Normalize static data.
        train_ind = self.splits['train']
        means = demo[train_ind].mean(axis=0, keepdims=True)
        stds = demo[train_ind].std(axis=0, keepdims=True)
        stds = (stds==0) + (stds>0)*stds
        demo = (demo-means)/stds
        self.args.logger.write('# static features: '+str(D))
        # to save
        self.demo = demo
        self.args.D = D
        return data


    def get_batch(self, ind=None):
        if ind is None:
            ind = self.train_cycler.get_batch_ind()
        if self.args.model_type in ['strats', 'istrats']:
            return self.get_batch_strats(ind)
        elif self.args.model_type=='grud':
            return self.get_batch_grud(ind)
        elif self.args.model_type=='interpnet':
            return self.get_batch_interpnet(ind)
        elif self.args.model_type in ['gru', 'tcn', 'sand']:
            return {'ts':torch.FloatTensor(self.X[ind]),
                    'demo':torch.FloatTensor(self.demo[ind]),
                    'labels':torch.FloatTensor(self.y[ind])}


    def get_batch_grud(self, ind):
        deltas = [self.deltas[i] for i in ind]
        values = [self.values[i] for i in ind]
        masks = [self.mask[i] for i in ind]
        num_timestamps = np.array(list(map(len, deltas)))
        max_timestamps = max(num_timestamps)
        pad_lens = max_timestamps-num_timestamps
        V = self.args.V
        pad_mats = [np.zeros((l,V)) for l in pad_lens]
        deltas = torch.FloatTensor(np.stack([np.concatenate((delta,pad), axis=0)
                                    for delta,pad in zip(deltas,pad_mats)]))
        values = torch.FloatTensor(np.stack([np.concatenate((delta,pad), axis=0)
                                    for delta,pad in zip(values,pad_mats)]))
        masks = torch.FloatTensor(np.stack([np.concatenate((delta,pad), axis=0)
                                    for delta,pad in zip(masks,pad_mats)]))
        return {'delta_t':deltas, 'x_t':values, 'm_t':masks,
                'seq_len':torch.LongTensor(num_timestamps),
                'demo':torch.FloatTensor(self.demo[ind]),
                'labels':torch.FloatTensor(self.y[ind])}

    def get_batch_interpnet(self, ind):
        times = [self.times[i] for i in ind]
        values = [self.values[i] for i in ind]
        masks = [self.mask[i] for i in ind]
        hmasks = [self.holdout_masks[i] for i in ind]

        num_timestamps = np.array(list(map(len, times)))
        max_timestamps = max(num_timestamps)
        pad_lens = max_timestamps-num_timestamps
        V = self.args.V
        pad_mats = [np.zeros((l,V)) for l in pad_lens]
        hmasks = torch.FloatTensor(np.stack([np.concatenate((delta,pad), axis=0)
                                    for delta,pad in zip(hmasks,pad_mats)]))
        values = torch.FloatTensor(np.stack([np.concatenate((delta,pad), axis=0)
                                    for delta,pad in zip(values,pad_mats)]))
        masks = torch.FloatTensor(np.stack([np.concatenate((delta,pad), axis=0)
                                    for delta,pad in zip(masks,pad_mats)]))
        times = torch.FloatTensor([t+[0]*p for t,p in zip(times, pad_lens)])
        return {'t':times, 'x':values, 'm':masks, 'h':hmasks,
                'demo':torch.FloatTensor(self.demo[ind]),
                'labels':torch.FloatTensor(self.y[ind])}


    def get_batch_strats(self, ind):
        demo = torch.FloatTensor(self.demo[ind]) # N,D
        num_obs = [len(self.values[i]) for i in ind]
        max_obs = max(num_obs)
        pad_lens = max_obs-np.array(num_obs)
        values = [self.values[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        times = [self.times[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        varis = [self.varis[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        values, times = torch.FloatTensor(values), torch.FloatTensor(times)
        varis = torch.IntTensor(varis)
        obs_mask = [[1]*l1+[0]*l2 for l1,l2 in zip(num_obs,pad_lens)]
        obs_mask = torch.IntTensor(obs_mask)
        return {'values':values, 'times':times, 'varis':varis,
                'obs_mask':obs_mask, 'demo':demo,
                'labels':torch.FloatTensor(self.y[ind])}
