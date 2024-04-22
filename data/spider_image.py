import argparse
import requests
import os
import csv


def get_item_image(dataset):
    csv_file_path = os.path.join('./data-clip/', dataset + '/' + dataset + '_url.csv')
    output_folder = os.path.join('./data-clip/', dataset + '/images')
    os.makedirs(output_folder, exist_ok=True)
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)    
        for row in reader:
            item_id = row['itemID']
            imurl = row['imUrl']
            if int(item_id) < 1230:
                continue
            try:
                headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400'}
                re = requests.get(imurl, headers=headers)
                print(re.status_code)
                path = output_folder + '/' + item_id + '.jpg'
                with open(path, 'wb') as f:
                    for chunk in re.iter_content(chunk_size=128):
                        f.write(chunk)
                print(f'Successfully downloaded {item_id}.jpg')
            except Exception as e:
                print(f'Error downloading {item_id}: {str(e)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='clothing', help='name of dataset')
    args = parser.parse_args()
    get_item_image(args.dataset)
