import json

import pandas as pd
import requests


def get_data(year, widget):
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Host': 'apidatos.ree.es'
    }
    url = f'https://apidatos.ree.es/en/datos/generacion/{widget}'
    params = {
        'time_trunc': 'month',
        'tecno_select': 'all',
        'start_date': f'{year}-01-01T00:00+01',
        'end_date': f'{year}-12-31T23:59+01'
    }

    print(f'[*] Fetching data for {year}...')
    r = requests.get(url, params, headers=headers)
    return json.loads(r.text)


def parse_data(data):
    technology_types = []
    dates = []
    values = []

    for tech in data['included']:
        tech_name = tech['type']

        for dated_value in tech['attributes']['values']:
            technology_types.append(tech_name)
            dates.append(dated_value['datetime'])
            values.append(dated_value['value'])

    return technology_types, dates, values


# Widget (see https://www.ree.es/en/apidatos) - Magnitude pairs
datasets = [
    ('estructura-generacion', 'generated_energy'),
    ('potencia-instalada', 'installed_power_capacity')
]

if __name__ == '__main__':
    for widget, magnitude in datasets:
        data = {
            'technology_type': [],
            'date': [],
            magnitude: []
        }

        for year in range(2015, 2023):
            types, dates, values = parse_data(get_data(year, widget))
            data['technology_type'].extend(types)
            data['date'].extend(dates)
            data[magnitude].extend(values)

        df = pd.DataFrame(data)
        df.to_csv(f'../data/{magnitude}.csv', index=False)
