import requests
import json
import pandas as pd
import numpy as np
import math

#state ids for getting features
'''
Training
New Hampshire: 33
Iowa: 19
Nevada: 32
South Carolina: 45

Testing
Texas: 48
California: 06

Wisconsin: 55
Pennsylvania: 42
Michigan: 26
'''
api_key = 'd37d7cef862aa62b780a8e3e663991e23d4d25e3'

rep_dem_url = 'https://www.politico.com/election-results/2018/iowa/county.json'

pres_res_url = 'https://raw.githubusercontent.com/tonmcg/US_County_Level_Election_Results_08-16/master/2016_US_County_Level_Presidential_Results.csv'

midterm_url = 'https://int.nyt.com/applications/elections/2014/data/2014-11-04/supermap/governor.json'

#feature selection and meaning
vars_dict = {
    'B01001_001E': 'total',
    #'B01001_002E': 'males',
    'B01001_007E': 'males18-19',
    'B01001_008E': 'males20',
    'B01001_009E': 'males21',
    'B01001_010E': 'males22-24',
    'B01001_011E': 'males25-29',
    'B01001_012E': 'males30-34',
    'B01001_013E': 'males35-39',
    'B01001_014E': 'males40-44',
    'B01001_015E': 'males45-49',
    'B01001_016E': 'males50-54',
    'B01001_017E': 'males55-59',
    'B01001_018E': 'males60-61',
    'B01001_019E': 'males62-64',
    'B01001_020E': 'males65-66',
    'B01001_021E': 'males67-69',
    'B01001_022E': 'males70-74',
    'B01001_023E': 'males75-79',
    'B01001_024E': 'males80-84',
    'B01001_025E': 'males>85',
    #'B01001_026E': 'females',
    'B01001_031E': 'females18-19',
    'B01001_032E': 'females20',
    'B01001_033E': 'females21',
    'B01001_034E': 'females22-24',
    'B01001_035E': 'females25-29',
    'B01001_036E': 'females30-34',
    'B01001_037E': 'females35-39',
    'B01001_038E': 'females40-44',
    'B01001_039E': 'females45-49',
    'B01001_040E': 'females50-54',
    'B01001_041E': 'females55-59',
    'B01001_042E': 'females60-61',
    'B01001_043E': 'females62-64',
    'B01001_044E': 'females65-66',
    'B01001_045E': 'females67-69',
    'B01001_046E': 'females70-74',
    'B01001_047E': 'females75-79',
    'B01001_048E': 'females80-84',
    'B01001_049E': 'females>85',
    'B01001A_002E': 'white_males',
    'B01001A_017E': 'white_females',
    'B01001B_002E': 'black_males',
    'B01001B_017E': 'black_females',
    'B01001C_002E': 'ai_males',
    'B01001C_017E': 'ai_females',
    'B01001D_002E': 'asian_males',
    'B01001D_017E': 'asian_females',
    'B01001E_002E': 'hawaiian_males',
    'B01001E_017E': 'hawaiian_females',
    'B01001F_002E': 'other_males',
    'B01001F_017E': 'other_females',
    'B01001G_002E': 'multi_males',
    'B01001G_017E': 'multi_females',
    'B01001I_002E': 'latino_males',
    'B01001I_017E': 'latino_females',
    #'B01002_002E': 'med_age_males',
    #'B01002_003E': 'med_age_females',
    #'B01002A_002E': 'med_age_males_w',
    #'B01002A_003E': 'med_age_females_w',
    #'B01002B_002E': 'med_age_males_b',
    #'B01002B_003E': 'med_age_females_b',
    #'B01002C_002E': 'med_age_males_ai',
    #'B01002C_003E': 'med_age_females_ai',
    #'B01002D_002E': 'med_age_males_a',
    #'B01002D_003E': 'med_age_females_a',
    #'B01002F_002E': 'med_age_males_o',
    #'B01002F_003E': 'med_age_females_o',
    #'B01002E_002E': 'med_age_males_h',
    #'B01002E_003E': 'med_age_females_h',
    #'B01002G_002E': 'med_age_males_m',
    #'B01002G_003E': 'med_age_females_m',
    #'B01002I_002E': 'med_age_males_l',
    #'B01002I_003E': 'med_age_females_l',
    'B05001_002E': 'us_born',
    'B05001_003E': 'us_island_born',
    'B05001_004E': 'us_par_abroad',
    'B05001_005E': 'naturalized',
    'B05001_006E': 'not_citizen',
    #'B05002_003E': 'born_in_state',
    #'B05002_005E': 'born_ne',
    #'B05002_006E': 'born_mw',
    #'B05002_007E': 'born_s',
    #'B05002_008E': 'born_w',
    'B05002_009E': 'born_outside_citizen',
    'B05002_013E': 'born_foreign',
    'B06008_002E': 'never_married',
    'B06008_003E': 'married_separated',
    'B06008_004E': 'divorced',
    'B06008_005E': 'separated',
    'B06008_006E': 'widowed',
    'B06009_002E': '<highschool',
    'B06009_003E': 'highschool',
    'B06009_004E': '<college',
    'B06009_005E': 'bachelor',
    'B06009_006E': 'graduate',
    #'B06012_002E': 'poverty100',
    #'B06012_003E': 'poverty100-149',
    #'B06012_004E': 'poverty>150',
    #'B07001_001E': 'geo_mobility',
    #'B08006_002E': 'drove_to_work',
    #'B08006_008E': 'public_transit',
    #'B08006_014E': 'bike',
    #'B08006_015E': 'walk',
    #'B08006_016E': 'taxi_motorcylce_etc',
    #B08006_017E': 'work_from_home',
    #'B08007_002E': 'work_instate',
    #'B08007_003E': 'work_incounty',
    #'B08007_004E': 'work_outcounty',
    #'B08007_005E': 'work_outstate',
    #'B08008_002E': 'living_in_place',
    #'B08014_002E': 'no_vehicles',
    #'B08014_003E': '1_vehicles',
    #'B08014_004E': '2_vehicles',
    #'B08014_005E': '3_vehicles',
    #'B08014_006E': '4_vehicles',
    #'B08014_007E': '5+_vehicles',
    'B08016_002E': 'live_in_princ_city',
    'B08016_013E': 'live_outside_city',
    'B09002_001E': 'have_children',
    'B09010_002E': 'supplemental_income',
    #'B09021_002E': 'lives_alone18+',
    #'B09021_003E': 'spouse18+',
    #'B09021_004E': 'partner18+',
    #'B09021_005E': 'child18+',
    #'B09021_006E': 'other_family18+',
    #'B09021_007E': 'others18+',
    'B11001_002E': 'family_households',
    #'B12001_003E': 'male_never_married',
    #'B12001_004E': 'male_married',
    #'B12001_009E': 'male_widowed',
    #'B12001_010E': 'male_divorced',
    #'B12001_012E': 'female_never_married',
    #'B12001_013E': 'female_married',
    #'B12001_018E': 'female_widowed',
    #'B12001_019E': 'female_divorced',
    #'B13002_002E': 'women_had_baby_past_year',
    'B14002_003E': 'male_school_enrollment',
    'B14002_027E': 'female_school_enrollment',
    #'B14003_003E': 'male_public_school',
    #'B14003_012E': 'male_private_school',
    #'B14003_031E': 'female_public_school',
    #'B14003_040E': 'female_private_school',
    'B19001_002E': 'income<10k',
    'B19001_003E': 'income10-15k',
    'B19001_004E': 'income15-20k',
    'B19001_005E': 'income20-25k',
    'B19001_006E': 'income25-30k',
    'B19001_007E': 'income30-35k',
    'B19001_008E': 'income35-40k',
    'B19001_009E': 'income40-45k',
    'B19001_010E': 'income45-50k',
    'B19001_011E': 'income50-60k',
    'B19001_012E': 'income60-75k',
    'B19001_013E': 'income75-100k',
    'B19001_014E': 'income100-125k',
    'B19001_015E': 'income125-150k',
    'B19001_016E': 'income150-200k',
    'B19001_017E': 'income>200k'
}

#columns of the final preprocessed dataframe
columns = ['%white_male', '%white_female', '%black_male', '%black_female', '%american_indian_male', '%american_indian_female', '%asian_male', '%asian_female', '%hawaiin/pacific_islander_male', '%hawaiin/pacific_islander_female', '%other_male', '%other_female', '%multi_male', '%multi_female', '%latino_male', '%latino_female', '%us_born', '%us_island_born', '%us_parents_abroad', '%naturalized', '%not_citizen', '%born_outside_us_citizen', '%born_foreign', '%never_married', '%married_separated', '%divorced', '%separated', '%widowed', '%<highschool', '%highschool', '%<college', '%bachelor', '%graduate', '%live_in_principal_city', '%live_outside_principal_city', '%have_children', '%receive_supplemental_income', '%has_family_household', '%male_school_enroll', '%female_school_enroll', '%income<10K', '%income>200K', '%males_18_35', '%males_35_65', '%males_65_plus', '%females_18_35', '%females_35_65', '%females_65_plus', '%income_10-40K', '%income_40-75K', '%income_75-125', '%income_125-200']


vars_keys = list(vars_dict.keys())

#gets the input features for a given state and year
def get_inputs(state_id, year):
    results = []

    #state_id needs to be exactly 2 digits
    if state_id < 10:
        state_id = '0' + str(state_id)

    #putting features in proper request format
    #can only request 50 at a time
    for i in range(0, len(vars_keys), 50):
        vars = ''
        for k in vars_keys[i: i+50]:
            vars += k + ','
        vars = vars[:-1]

        url = f'https://api.census.gov/data/{year}/acs/acs5?get={vars}&for=county:*&in=state:{state_id}&key={api_key}'
        response = requests.get(url)
        json_data = response.json()

        columns = json_data[0]
        json_data = json_data[1:]

        new_df = pd.DataFrame(json_data, columns = columns)
        new_df = new_df.drop(columns = 'state')
        results.append(new_df)

    #combine the results of the different calls for features
    inputs = None
    for i, df in enumerate(results):
        if i == 0:
            inputs = df
        else:
            inputs = inputs.merge(df, on='county', how = 'left')

    #get fips by combining state id and county id
    inputs['fips'] = inputs['county'].apply(lambda x: int(x) + 1000 * int(state_id))
    inputs = inputs.drop(columns = 'county')
    inputs = inputs.sort_values(['fips'])
    return inputs

#gets the percent that voted democrat for 2018 governor election
#to be added to input features
def get_dem(url):
    dem_df = pd.read_json(url)
    dem_results = dem_df[dem_df['party'] == 'Dem']
    dem_results['fips'] = dem_results['fipscode']

    dem_avg = dem_results.groupby('fips').mean()
    dem_avg = dem_avg.sort_values(['fips'])

    return dem_avg['votepct']


nh_url = 'https://int.nyt.com/applications/elections/2020/data/api/2020-02-11/new-hampshire/president/democrat.json'
iowa_url = 'https://int.nyt.com/applications/elections/2020/data/api/2020-02-03/iowa/president/democrat.json'
nevada_url = 'https://int.nyt.com/applications/elections/2020/data/api/2020-02-22/nevada/president/democrat.json'
sc_url = 'https://int.nyt.com/applications/elections/2020/data/api/2020-02-29/south-carolina/president/democrat.json'

ca_url = 'https://int.nyt.com/applications/elections/2020/data/api/2020-03-03/california.json'
texas_url = 'https://int.nyt.com/applications/elections/2020/data/api/2020-03-03/texas.json'

#takes the final align results and finds the candidate with the highest number of votes
def get_winners(results_df):
    results_df = results_df
    results = results_df.drop(columns = ['fips', 'name', 'precincts', 'votes', 'reporting'])
    results = results.to_numpy()
    winners = np.nanargmax(results, axis = 1)

    winners = pd.DataFrame(winners, columns = ['winner'])
    fips_winner = pd.concat([results_df['fips'], winners], axis = 1)

    return pd.DataFrame(fips_winner)

#gets final align results of all of the candidates for the primaries
def get_results(url):
    caucus_df = pd.read_json(url)
    caucus_dict = caucus_df['data'][0][0]

    county_dicts = caucus_dict['counties']

    county_res = []

    for county in county_dicts:
        try:
            id = float(county['fips'])

            for k, v in county['results'].items():
                county['results_' + k] = v

            '''
            for k, v in county['results_align1'].items():
                county['align1_' + k] = v

            for k, v in county['results_alignfinal'].items():
                county['alignf_' + k] = v
            '''

            res = county.pop('results')


            county.pop('results_align1', None)
            county.pop('results_alignfinal', None)
            county.pop('votes_alignfinal', None)
            county.pop('votes_align1', None)

            county['fips'] = int(county['fips'])
            county_res.append(county)

        except:
            pass

    county_res = pd.DataFrame(county_res).sort_values(['fips'])
    #print(county_res)
    return county_res


#combines some of the columns into broader ranges to reduce number of parameters
#specifically, combines some of the age groups and income groups
def clean_inputs(input_df):
    input_df = input_df.astype(float)
    filter_columns = input_df.columns[(input_df < 0).any()]

    input_df = input_df.drop(columns = filter_columns)


    males_18_35 = ['B01001_007E', 'B01001_008E', 'B01001_009E', 'B01001_010E', 'B01001_011E', 'B01001_012E']
    males_35_65 = ['B01001_013E', 'B01001_014E', 'B01001_015E', 'B01001_016E', 'B01001_017E', 'B01001_018E', 'B01001_019E']
    males_65_plus = ['B01001_020E', 'B01001_021E', 'B01001_022E', 'B01001_023E', 'B01001_024E', 'B01001_025E']

    females_18_35 = ['B01001_031E', 'B01001_032E', 'B01001_033E', 'B01001_034E', 'B01001_035E', 'B01001_036E']
    females_35_65 = ['B01001_037E', 'B01001_038E', 'B01001_039E', 'B01001_040E', 'B01001_041E', 'B01001_042E', 'B01001_043E']
    females_65_plus = ['B01001_044E', 'B01001_045E', 'B01001_046E', 'B01001_047E', 'B01001_048E', 'B01001_049E']

    income_10_40 = ['B19001_003E', 'B19001_004E', 'B19001_005E', 'B19001_006E', 'B19001_007E', 'B19001_008E']
    income_40_75 = ['B19001_009E', 'B19001_010E', 'B19001_011E', 'B19001_012E']
    income_75_125 = ['B19001_013E', 'B19001_014E']
    income_125_200 = ['B19001_015E', 'B19001_016E']

    for i, col in enumerate(males_18_35):
        if i == 0:
            col_sum = input_df.loc[:, col].astype(int)
        else:
            col_sum = col_sum + input_df.loc[:, col].astype(int)
    input_df['males_18_35'] = col_sum
    input_df = input_df.drop(columns = males_18_35)


    for i, col in enumerate(males_35_65):
        if i == 0:
            col_sum = input_df.loc[:, col].astype(int)
        else:
            col_sum = col_sum + input_df.loc[:, col].astype(int)
    input_df['males_35_65'] = col_sum
    input_df = input_df.drop(columns = males_35_65)


    for i, col in enumerate(males_65_plus):
        if i == 0:
            col_sum = input_df.loc[:, col].astype(int)
        else:
            col_sum = col_sum + input_df.loc[:, col].astype(int)

    input_df['males_65_plus'] = col_sum
    input_df = input_df.drop(columns = males_65_plus)


    for i, col in enumerate(females_18_35):
        if i == 0:
            col_sum = input_df.loc[:, col].astype(int)
        else:
            col_sum = col_sum + input_df.loc[:, col].astype(int)
    input_df['females_18_35'] = col_sum
    input_df = input_df.drop(columns = females_18_35)


    for i, col in enumerate(females_35_65):
        if i == 0:
            col_sum = input_df.loc[:, col].astype(int)
        else:
            col_sum = col_sum + input_df.loc[:, col].astype(int)
    input_df['females_35_65'] = col_sum
    input_df = input_df.drop(columns = females_35_65)


    for i, col in enumerate(females_65_plus):
        if i == 0:
            col_sum = input_df.loc[:, col].astype(int)
        else:
            col_sum = col_sum + input_df.loc[:, col].astype(int)

    input_df['females_65_plus'] = col_sum
    input_df = input_df.drop(columns = females_65_plus)


    for i, col in enumerate(income_10_40):
        if i == 0:
            col_sum = input_df.loc[:, col].astype(int)
        else:
            col_sum = col_sum + input_df.loc[:, col].astype(int)
    input_df['income_10_40'] = col_sum
    input_df = input_df.drop(columns = income_10_40)

    for i, col in enumerate(income_40_75):
        if i == 0:
            col_sum = input_df.loc[:, col].astype(int)
        else:
            col_sum = col_sum + input_df.loc[:, col].astype(int)
    input_df['income_40_75'] = col_sum
    input_df = input_df.drop(columns = income_40_75)

    for i, col in enumerate(income_75_125):
        if i == 0:
            col_sum = input_df.loc[:, col].astype(int)
        else:
            col_sum = col_sum + input_df.loc[:, col].astype(int)
    input_df['income_75_125'] = col_sum
    input_df = input_df.drop(columns = income_75_125)

    for i, col in enumerate(income_125_200):
        if i == 0:
            col_sum = input_df.loc[:, col].astype(int)
        else:
            col_sum = col_sum + input_df.loc[:, col].astype(int)
    input_df['income_125_200'] = col_sum
    input_df = input_df.drop(columns = income_125_200)


    return input_df


#gets the governor election results by county for iowa, michigan, Pennsylvania, and Wisconsin
#url is the link to the data
#margin indicates whether the winner column should be 0,1, 0 meaning democrats win,
#   or the percent of the votes that were for the democrats
#returns the fips, county name, and winner for the counties in a dataframe
# democrat is 1, republican is 2 in this dataset
def midterm_elections(url, margin = False):
    df = pd.read_json(url)
    states = df.loc[:, ['ia-governor-2014-general', 'pa-governor-2014-general', 'mi-governor-2014-general', 'wi-governor-2014-general']]
    states = states.iloc[1, :]

    counties = pd.DataFrame([], columns = ['fips', 'winner'])

    for state in states:
        for county in state:
            fips_winner = {'fips': county['id'], 'county': county['name']}
            if not margin:
                winner = int(list(county.keys())[5][0]) - 1
                fips_winner['winner'] = winner
            else:
                if int(list(county.keys())[5][0]) == 1:
                    num_dem = int(list(county.values())[5])
                else:
                    num_dem = int(list(county.values())[6])
                num_total = int(county['votes'])
                fips_winner['winner'] = (num_dem / num_total)


            counties = counties.append(fips_winner, ignore_index = True)

    counties = counties.sort_values(['fips'])

    if not margin:
        counties.to_csv('data/gov_winners.csv')
    else:
        counties.to_csv('data/gov_winners_margin.csv')

    return counties


#gets the presidential election results by county for iowa, michigan, Pennsylvania, and Wisconsin
#url is the link to the data
#margin indicates whether the winner column should be 0,1, 0 meaning democrats win,
#   or the percent of the votes that were for the democrats
#returns the fips, winner, county, and state for the counties in a dataframe
def pres_elections(url, margin = False):
    df = pd.read_csv(url)
    pres_df = df.loc[:, ['combined_fips', 'per_dem', 'per_gop', 'county_name', 'state_abbr']]
    if not margin:
        pres_df['winner'] = pres_df.loc[:, ['per_dem', 'per_gop']].apply(lambda x: int(x.iloc[0] < x.iloc[1]), axis = 1)
    else:
        pres_df['winner'] = pres_df['per_dem']
    pres_df['fips'] = pres_df['combined_fips']

    pres_df = pres_df[(pres_df['fips'].apply(lambda x: math.floor(x/1000) in [55, 42, 26, 19]))]
    pres_df['county'] = pres_df['county_name'].apply(lambda x: x.replace('County', '').strip())


    fips_winner = pres_df.loc[:, ['fips', 'winner', 'county', 'state_abbr']]
    fips_winner = fips_winner.sort_values(['fips'])
    fips_winner = fips_winner.reset_index(drop = True)

    if not margin:
        fips_winner.to_csv('data/pres_winners.csv')
    else:
        fips_winner.to_csv('data/pres_winners_margin.csv')

    return fips_winner



def main():
    '''
    #primary data processing
    #get primary results
    iowa_res = get_results(iowa_url)
    nh_res = get_results(nh_url)
    sc_res = get_results(sc_url)
    nevada_res = get_results(nevada_url)

    ca_res = get_results(ca_url)
    texas_res = get_results(texas_url)

    #combine all the data (so all candidates are accounted even if they did not run in all primaries)
    all_res = pd.concat([iowa_res, nh_res, sc_res, nevada_res, ca_res, texas_res], ignore_index = True)

    #drop the candidates that only appeared in 10 counties' results
    drop_columns = all_res.columns[all_res.count() <= 10]
    all_res = all_res.drop(columns = drop_columns)

    #print(all_res)

    #separate out the testing and trainging results
    testing = all_res[(all_res['fips'].apply(lambda x: math.floor(x/1000)) == 48) | (all_res['fips'].apply(lambda x: math.floor(x/1000)) == 6)]
    training = all_res[(all_res['fips'].apply(lambda x: math.floor(x/1000)) != 48) & (all_res['fips'].apply(lambda x: math.floor(x/1000)) != 6)]

    testing = testing.sort_values(['fips']).reset_index(drop = True)
    training = training.sort_values(['fips'])

    testing = get_winners(testing)
    training = get_winners(training)

    #save the results for easy access in other files
    testing.to_csv('data/testing_results.csv')
    training.to_csv('data/training_results.csv')



    #get the states demographics/inputs
    iowa_inp = clean_inputs(get_inputs(19, 2018))
    nh_inp = clean_inputs(get_inputs(33, 2018))
    sc_inp = clean_inputs(get_inputs(45, 2018))
    nev_inp = clean_inputs(get_inputs(32, 2018))



    #remove fips and total population columns, rows ordered by fips
    train_inputs_pop = iowa_inp['B01001_001E'].to_numpy()
    train_inputs = iowa_inp.drop(columns = ['fips', 'B01001_001E']).to_numpy()

    #normalize columns by total population
    train_inputs = np.transpose(np.transpose(train_inputs) / train_inputs_pop)

    #return to dataframe and give new column names
    iowa_inp_win = pd.DataFrame(train_inputs, columns = columns)

    #add winner column - mainly to make it easier to run in r
    iowa_res = get_winners(iowa_res)
    iowa_inp_win = pd.concat([iowa_inp, iowa_res['winner']], axis = 1)

    #save iowa data for separate examination
    #iowa_inp_win.to_csv('data/iowa_inputs.csv')
    #iowa_res.to_csv('data/iowa_results.csv')


    #combine all the trainging and testing states respectively and save data
    all_train_inp = pd.concat([iowa_inp, nh_inp, sc_inp, nev_inp]).sort_values(['fips']).reset_index(drop = True)

    texas_inp = clean_inputs(get_inputs(48, 2018))
    ca_inp = clean_inputs(get_inputs(6, 2018))

    all_test_inp = pd.concat([texas_inp, ca_inp]).sort_values(['fips']).reset_index(drop = True)

    #test iowa 2018 governor results
    #dem_table = get_dem(rep_dem_url)
    #dem_table.to_csv('data/iowa_dem.csv')

    #get total population column
    train_inputs_pop = all_train_inp['B01001_001E'].to_numpy()
    test_inputs_pop = all_test_inp['B01001_001E'].to_numpy()

    #remove fips and total population columns, rows ordered by fips
    train_inputs = all_train_inp.drop(columns = ['fips', 'B01001_001E']).to_numpy()
    test_inputs = all_test_inp.drop(columns = ['fips', 'B01001_001E']).to_numpy()

    #normalize columns by total population
    train_inputs = np.transpose(np.transpose(train_inputs) / train_inputs_pop)
    test_inputs = np.transpose(np.transpose(test_inputs) / test_inputs_pop)

    #return to dataframe and give new column names
    all_train_inp = pd.DataFrame(train_inputs, columns = columns)
    all_test_inp = pd.DataFrame(test_inputs, columns = columns)

    #add winner column - mainly to make it easier to run in r
    #all_train_inp = pd.concat([all_train_inp, training['winner']], axis = 1)
    #all_test_inp = pd.concat([all_test_inp, testing.loc[:, ['winner']]], axis = 1)
    all_test_inp = all_test_inp.dropna(how = 'all')


    all_train_inp.to_csv('data/primary_training_inputs.csv', index = False)
    all_test_inp.to_csv('data/primary_testing_inputs.csv', index = False)
    '''

    #get the inputs for each state and year
    ia_inp_2014 = clean_inputs(get_inputs(19, 2014))
    pa_inp_2014 = clean_inputs(get_inputs(42, 2014))
    mi_inp_2014 = clean_inputs(get_inputs(26, 2014))
    wi_inp_2014 = clean_inputs(get_inputs(55, 2014))

    ia_inp_2016 = clean_inputs(get_inputs(19, 2016))
    pa_inp_2016 = clean_inputs(get_inputs(42, 2016))
    mi_inp_2016 = clean_inputs(get_inputs(26, 2016))
    wi_inp_2016 = clean_inputs(get_inputs(55, 2016))

    #combine the different states data
    train_2014 = pd.concat([ia_inp_2014, mi_inp_2014, pa_inp_2014, wi_inp_2014])
    test_2016 = pd.concat([ia_inp_2016, mi_inp_2016, pa_inp_2016, wi_inp_2016])

    #get results/labels for data
    county_winners = midterm_elections(midterm_url)
    pres_winners = pres_elections(pres_res_url)


    #get total population column
    train_inputs_pop = train_2014['B01001_001E'].to_numpy()
    test_inputs_pop = test_2016['B01001_001E'].to_numpy()

    #remove fips and total population columns, rows ordered by fips
    train_inputs = train_2014.drop(columns = ['fips', 'B01001_001E']).to_numpy()
    test_inputs = test_2016.drop(columns = ['fips', 'B01001_001E']).to_numpy()

    #normalize columns by total population
    train_inputs = np.transpose(np.transpose(train_inputs) / train_inputs_pop)
    test_inputs = np.transpose(np.transpose(test_inputs) / test_inputs_pop)

    #return to dataframe and give new column names
    train_2014 = pd.DataFrame(train_inputs, columns = columns)
    test_2016 = pd.DataFrame(test_inputs, columns = columns)

    #add winner column - mainly to make it easier to run in r
    train_2014 = pd.concat([train_2014, county_winners['winner']], axis = 1)
    test_2016 = pd.concat([test_2016, pres_winners.loc[:, ['winner']]], axis = 1)

    #save the data for easy access in other files
    train_2014.to_csv('data/2014_data_r.csv', index = False)
    test_2016.to_csv('data/2016_data_r.csv', index = False)

    #get the county, state, and winner for map visualization
    #maps_df = pres_winners.loc[:, ['county', 'state_abbr', 'winner']]
    #maps_df.to_csv('data/maps.csv', index = False)




if __name__ == '__main__':
    main()
