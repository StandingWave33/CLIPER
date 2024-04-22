import os, csv
import pandas as pd
import argparse 
from collections import Counter
import numpy as np
import gzip, json, array
import torch
from PIL import Image
# import clip
from Long_CLIP.model import longclip
from spider_image import get_item_image

min_u_num, min_i_num = 5, 5
learner_id, course_id, tmstmp_str = 'userID', 'itemID', 'timestamp'
def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10).decode('UTF-8')
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)


def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


def get_illegal_ids_by_inter_num(df, field, max_num=None, min_num=None):
    if field is None:
        return set()
    if max_num is None and min_num is None:
        return set()
    max_num = max_num or np.inf
    min_num = min_num or -1
    ids = df[field].values
    inter_num = Counter(ids)
    ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}
    print(f'{len(ids)} illegal_ids_by_inter_num, field={field}')
    return ids


def filter_by_k_core(df):
    while True:
        ban_users = get_illegal_ids_by_inter_num(df, field=learner_id, max_num=None, min_num=min_u_num)
        ban_items = get_illegal_ids_by_inter_num(df, field=course_id, max_num=None, min_num=min_i_num)
        if len(ban_users) == 0 and len(ban_items) == 0:
            return

        dropped_inter = pd.Series(False, index=df.index)
        if learner_id:
            dropped_inter |= df[learner_id].isin(ban_users)
        if course_id:
            dropped_inter |= df[course_id].isin(ban_items)
        print(f'{len(dropped_inter)} dropped interactions')
        df.drop(df.index[dropped_inter], inplace=True)


def rating2inter():
    # Load data
    df = pd.read_csv('ratings_' + args.dataset +'.csv', names=['userID', 'itemID', 'rating', 'timestamp'], header=None)
    print(f'shape: {df.shape}')
    k_core = 5
    df.dropna(subset=[learner_id, course_id, tmstmp_str], inplace=True)
    df.drop_duplicates(subset=[learner_id, course_id, tmstmp_str], inplace=True)
    print(f'After dropped: {df.shape}')
    filter_by_k_core(df)
    print(f'k-core shape: {df.shape}')
    print(f'shape after k-core: {df.shape}')

    # Re-index
    df.reset_index(drop=True, inplace=True)
    i_mapping_file = 'i_id_mapping.csv'
    u_mapping_file = 'u_id_mapping.csv'
    splitting = [0.8, 0.1, 0.1]
    uid_field, iid_field = learner_id, course_id
    uni_users = pd.unique(df[uid_field])
    uni_items = pd.unique(df[iid_field])
    # start from 0
    u_id_map = {k: i for i, k in enumerate(uni_users)}
    i_id_map = {k: i for i, k in enumerate(uni_items)}
    df[uid_field] = df[uid_field].map(u_id_map)
    df[iid_field] = df[iid_field].map(i_id_map)
    df[uid_field] = df[uid_field].astype(int)
    df[iid_field] = df[iid_field].astype(int)
    # dump
    rslt_dir = './'
    u_df = pd.DataFrame(list(u_id_map.items()), columns=['user_id', 'userID'])
    i_df = pd.DataFrame(list(i_id_map.items()), columns=['asin', 'itemID'])
    u_df.to_csv(os.path.join(rslt_dir, u_mapping_file), sep='\t', index=False)
    i_df.to_csv(os.path.join(rslt_dir, i_mapping_file), sep='\t', index=False)
    print(f'mapping dumped...')
    # =========2. splitting
    print(f'splitting ...')
    tot_ratio = sum(splitting)
    # remove 0.0 in ratios
    ratios = [i for i in splitting if i > .0]
    ratios = [_ / tot_ratio for _ in ratios]
    split_ratios = np.cumsum(ratios)[:-1]
    ts_id = 'timestamp'
    split_timestamps = list(np.quantile(df[ts_id], split_ratios))
    # get df training dataset unique users/items
    df_train = df.loc[df[ts_id] < split_timestamps[0]].copy()
    df_val = df.loc[(split_timestamps[0] <= df[ts_id]) & (df[ts_id] < split_timestamps[1])].copy()
    df_test = df.loc[(split_timestamps[1] <= df[ts_id])].copy()
    x_label, rslt_file = 'x_label', args.dataset + '-indexed.inter'
    df_train[x_label] = 0
    df_val[x_label] = 1
    df_test[x_label] = 2
    temp_df = pd.concat([df_train, df_val, df_test])
    temp_df = temp_df[[learner_id, course_id, 'rating', ts_id, x_label]]
    print(f'columns: {temp_df.columns}')
    temp_df.columns = [learner_id, course_id, 'rating', ts_id, x_label]
    temp_df.to_csv(os.path.join(rslt_dir, rslt_file), sep='\t', index=False)
    
    # reload
    indexed_df = pd.read_csv(rslt_file, sep='\t')
    print(f'shape: {indexed_df.shape}')
    u_uni = indexed_df[learner_id].unique()
    c_uni = indexed_df[course_id].unique()
    print(f'# of unique learners: {len(u_uni)}')
    print(f'# of unique courses: {len(c_uni)}')
    print('min/max of unique learners: {0}/{1}'.format(min(u_uni), max(u_uni)))
    print('min/max of unique courses: {0}/{1}'.format(min(c_uni), max(c_uni)))


def spiltting():
    rslt_file = args.dataset + '-indexed.inter'
    df = pd.read_csv(rslt_file, sep='\t')
    print(f'shape: {df.shape}')
    df = df.sample(frac=1).reset_index(drop=True)
    df.sort_values(by=['userID'], inplace=True) 

    uid_field, iid_field = 'userID', 'itemID'
    uid_freq = df.groupby(uid_field)[iid_field]
    u_i_dict = {}
    for u, u_ls in uid_freq:
        u_i_dict[u] = list(u_ls)
    u_i_dict

    new_label = []
    u_ids_sorted = sorted(u_i_dict.keys())
    for u in u_ids_sorted:
        items = u_i_dict[u]
        n_items = len(items)
        if n_items < 10:
            tmp_ls = [0] * (n_items - 2) + [1] + [2]
        else:
            val_test_len = int(n_items * 0.2)
            train_len = n_items - val_test_len
            val_len = val_test_len // 2
            test_len = val_test_len - val_len
            tmp_ls = [0] * train_len + [1] * val_len + [2] * test_len
        new_label.extend(tmp_ls)
    df['x_label'] = new_label
    new_labeled_file = rslt_file[:-6] + '.inter'
    df.to_csv(os.path.join('./', new_labeled_file), sep='\t', index=False)
    print('done!!!')

    # Reload
    indexed_df = pd.read_csv(new_labeled_file, sep='\t')
    print(f'shape: {indexed_df.shape}')
    u_id_str, i_id_str = 'userID', 'itemID'
    u_uni = indexed_df[u_id_str].unique()
    c_uni = indexed_df[i_id_str].unique()
    print(f'# of unique learners: {len(u_uni)}')
    print(f'# of unique courses: {len(c_uni)}')
    print('min/max of unique learners: {0}/{1}'.format(min(u_uni), max(u_uni)))
    print('min/max of unique courses: {0}/{1}'.format(min(c_uni), max(c_uni)))


def reindex_feat():
    i_id_mapping = 'i_id_mapping.csv'
    df = pd.read_csv(i_id_mapping, sep='\t')
    print(f'shape: {df.shape}')
    meta_file = 'meta_' + args.dataset + '.json.gz'
    print('0 Extracting U-I interactions.')
    meta_df = getDF(meta_file)
    print(f'Total records: {meta_df.shape}')

    # remapping
    map_dict = dict(zip(df['asin'], df['itemID']))
    meta_df['itemID'] = meta_df['asin'].map(map_dict)
    meta_df.dropna(subset=['itemID'], inplace=True)
    meta_df['itemID'] = meta_df['itemID'].astype('int64')
    meta_df.sort_values(by=['itemID'], inplace=True)
    print(f'shape: {meta_df.shape}')
    ori_cols = meta_df.columns.tolist()
    ret_cols = [ori_cols[-1]] + ori_cols[:-1]
    print(f'new column names: {ret_cols}')
    ret_df = meta_df[ret_cols]
    # dump
    ret_df.to_csv(os.path.join('./', args.dataset + '.csv'), index=False)
    image_url_df = ret_df[['itemID', 'asin', 'imUrl']]
    image_url_df.to_csv(os.path.join('./', args.dataset + '_url.csv'), index=False)
    print('done!')

    # Reload
    indexed_df = pd.read_csv(args.dataset + '.csv')
    print(f'shape: {indexed_df.shape}')
    i_uni = indexed_df['itemID'].unique()
    print(f'# of unique items: {len(i_uni)}')
    print('min/max of unique learners: {0}/{1}'.format(min(i_uni), max(i_uni)))


def feat_encoder():
    i_id, desc_str = 'itemID', 'description'
    meta_file = os.path.join('./', args.dataset + '.csv')
    df = pd.read_csv(meta_file)
    df.sort_values(by=[i_id], inplace=True)
    print('data loaded!')
    print(f'shape: {df.shape}')
    # sentences: title + brand + category + description | All have title + description
    title_na_df = df[df['title'].isnull()]
    print(title_na_df.shape)
    desc_na_df = df[df['description'].isnull()]
    print(desc_na_df.shape)
    na_df = df[df['description'].isnull() & df['title'].isnull()]
    print(na_df.shape)
    na3_df = df[df['description'].isnull() & df['title'].isnull() & df['brand'].isnull()]
    print(na3_df.shape)
    na4_df = df[df['description'].isnull() & df['title'].isnull() & df['brand'].isnull() & df['categories'].isnull()]
    print(na4_df.shape)

    df[desc_str] = df[desc_str].fillna(" ")
    df['title'] = df['title'].fillna(" ")
    df['brand'] = df['brand'].fillna(" ")
    df['categories'] = df['categories'].fillna(" ")

    sentences_tilte, sentences_brand, sentences_categories = [], [], []
    sentences_description, sentences_global = [], []
    title_count, brand_count, cate_count, description_count =0, 0, 0, 0
    for i, row in df.iterrows():
        cates = eval(row['categories'])                         # sen_catex
        cate_value = ''
        if isinstance(cates, list):
            for c in cates[0]:
                cate_value = cate_value + c + ' '
        sen_global = 'The product categories are {}.The product title is {}.The product brand is {}.' +\
            'The product description is as following:{}'.format(cate_value, row['title'], row['brand'], row[desc_str])
        sentences_global.append(sen_global)
        if len(cate_value)>1:
            sen_catex = 'The product categories are {}.'.format(cate_value)
            cate_count += 1
        else:
            sen_catex = 'The product categories are Missing.'
        sentences_categories.append(sen_catex)
        if len(row['title'])>1:
            sen_title = 'The product title is {}.'.format(row['title'])     # sen_title
            title_count += 1
        else:
            sen_title = 'The product title is Missing.'
        sentences_tilte.append(sen_title)
        if len(row['brand'])>1:
            print("{}, {}, {}".format(brand_count, row['brand'], len(row['brand'])))
            sen_brand = 'The product brand is {}.'.format(row['brand'])     # sen_brand
            brand_count += 1
        else:
            sen_brand = 'The product brand is Missing.'
        sentences_brand.append(sen_brand)
        if len(row[desc_str])>1:
            sen_description = 'The product description is {}'.format(row[desc_str]) # sen_description 
            sen_description = sen_description.replace('\n', ' ')
            description_count += 1
        else:
            sen_description = 'The product description is Missing.'
        sentences_description.append(sen_description)
    print("title_count: {}  brand_count:{}  cate_count:{}   description_count:{}".format(title_count, brand_count, cate_count, description_count))

    course_list = df[i_id].tolist()
    assert course_list[-1] == len(course_list) - 1

    # Encode text and img by CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = longclip.load("../Long_CLIP/checkpoints/longclip-L.pt", device=device)
    # model, preprocess = clip.load("ViT-L/14@336px", device=device)

    with torch.no_grad():
        all_text_features = []
        all_img_features = []
        print("Start to encode text")
        all_sentences = [sentences_tilte, sentences_brand, sentences_categories, sentences_description, sentences_global]
        sen_prompt = ['The product title is ','The product brand is ','The product categories are ','The product description is ', '']
        for index, sentences in enumerate(all_sentences):
            single_text = []
            for sen in sentences:
                try:
                    text = longclip.tokenize(sen).to(device)
                    text_features = model.encode_text(text)
                except:
                    print(sen)
                    max_length = 230
                    text_list = [sen[0:min(len(sen), max_length)]]
                    for i in range(1, len(sen)//max_length+1):
                        text_list.append(sen_prompt[index]+sen[max_length*i:min(len(sen), max_length*(i+1))])
                    text = longclip.tokenize(text_list).to(device)
                    text_features = model.encode_text(text)
                    text_features = torch.sum(text_features, dim=0)
                single_text.append(text_features.cpu().numpy())
            all_text_features.append(single_text)
        print("Start to encode image")
        url_df = pd.read_csv('./'+args.dataset + '_url.csv')
        not_exist = []
        for url_id in url_df['itemID']:
            try:
                image = Image.open('./images/'+str(url_id)+'.jpg')
            except:
                image = Image.open('./images/default.jpg')
                not_exist.append(url_id)
            image_input = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(image_input)
            all_img_features.append(image_features.cpu().numpy())
    print("Not exist:", not_exist)
    all_text_features = np.squeeze(np.array(all_text_features))
    all_img_features = np.squeeze(np.array(all_img_features))
    assert all_text_features.shape[1] == df.shape[0]
    assert all_img_features.shape[0] == df.shape[0]
    file_path = './'
    np.save(os.path.join(file_path, 'text_feat.npy'), all_text_features)
    np.save(os.path.join(file_path, 'image_feat.npy'), all_img_features)
    np.savetxt("missed_img_itemIDs.csv", not_exist, delimiter =",", fmt ='%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='clothing', help='name of dataset')
    args = parser.parse_args()
    data_path = os.chdir('/mnt/MMRec/data/' + args.dataset)
    rating2inter()
    spiltting()
    reindex_feat()
    get_item_image(args.dataset)
    feat_encoder()
   