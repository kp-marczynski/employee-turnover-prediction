import datetime
import re
from itertools import chain

import numpy as np
import pandas as pd


# number = re.compile('[\d,]+')


def get_first_number(val):
    val = str(val).lower()

    val = val.replace(",000", "000")
    change = 0
    if val.startswith("under") \
            or val.startswith("less than") \
            or val.startswith("fewer than") \
            or val.startswith("before") \
            or val.startswith("younger than"):
        change = -1
    elif val.startswith("more than") \
            or val.startswith("older than"):
        change = 1
    search = re.search(r'[\d\.\d]+', val)
    if search is None:
        return np.nan
    else:
        try:
            return float(search.group()) + change
        except:
            return 0


def dict_map(dict_to_use):
    def mapper(val):
        if not isinstance(val, str):
            val = str(val)
        if val.lower() in dict_to_use:
            return dict_to_use[val.lower()]
        else:
            return np.nan

    return mapper


def split_list(list_str):
    if pd.isnull(list_str):
        return []
    else:
        return [x.strip() for x in list_str.split(";")]


def userlist_to_cols(col):
    # col = col.where(col.notnull(), None)
    list_elems = col.astype('string').apply(split_list)
    categories = set(chain.from_iterable(list_elems.values))
    categories.discard('nan')
    categories = pd.Series(list(categories))

    def category_to_cols(category):
        return list_elems.map(lambda user_resp: category in user_resp).rename('%s_%s' % (col.name, category)).astype(
            'int8')

    new_cols = categories.map(category_to_cols)
    return pd.concat(new_cols.values, axis=1)


def timestr_to_number(timestr):
    if timestr == 'Noon':
        return 12
    elif timestr == 'Midnight':
        return 0
    else:
        return datetime.datetime.strptime(timestr, '%I:%M %p').hour


listvals = [
    'AdBlockerReasons',
    'AdsActions',
    'AuditoryEnvironment',
    'CommunicationTools',
    'CompanyType',
    'Containers',
    'CousinEducation',
    'DatabaseDesireNextYear',
    'DatabaseWorkedWith',
    'DeveloperType',
    'DevEnviron',
    'DevType',
    'EducationTypes',
    'EduOther',
    'ErgonomicDevices',
    'EthicsResponsible',
    'FrameworkDesireNextYear',
    'FrameworkWorkedWith',
    'HaveWorkedDatabase',
    'HaveWorkedFramework',
    'HaveWorkedLanguage',
    'HaveWorkedPlatform',
    'HopeFiveYears',
    'IDE',
    'ImportantBenefits',
    'JobFactors',
    'JobProfile',
    'LanguageDesireNextYear',
    'LanguageWorkedWith',
    'LastInt',
    'MainBranch',
    'MajorUndergrad',
    'Methodology',
    'MetricAssess',
    'MiscTechDesireNextYear',
    'MiscTechWorkedWith',
    'MobileDeveloperType',
    'NonDeveloperType',
    'OperatingSystem',
    'OpSys',
    'PlatformDesireNextYear',
    'PlatformWorkedWith',
    'Professional',
    'PurchaseHow',
    'Race',
    'SelfTaughtTypes',
    'SOVisitTo',
    'StackOverflowDevices',
    'UndergradMajor',
    'VersionControl',
    'WantWorkDatabase',
    'WantWorkFramework',
    'WantWorkLanguage',
    'WantWorkPlatform',
    'WebDeveloperType',
    'WebFrameDesireNextYear',
    'WebFrameWorkedWith',
    'WorkChallenge',
    'WorkLoc',
    'WorkPlan',
]

number_parses = [
    'Age',
    'Age1stCode',
    'CompanySize',
    'HoursComputer',
    'NumberMonitors',
    'OrgSize',
    'SOHowMuchTime',
    'StackOverflowJobsRecommend',
    'StackOverflowRecommend',
    'WakeTime',
    'YearsCode',
    'YearsCodedJob',
    'YearsCodedJobPast',
    'YearsCodePro',
    'YearsCoding',
    'YearsCodingProf',
    'YearsProgram',
]

agree_keys = [
    'AdsAgreeDisagree1',
    'AdsAgreeDisagree2',
    'AdsAgreeDisagree3',
    'AgreeDisagree1',
    'AgreeDisagree2',
    'AgreeDisagree3',
    'AnnoyingUI',
    'BoringDetails',
    'BuildingThings',
    'ChallengeMyself',
    'ChangeWorld',
    'CollaborateRemote',
    'CompetePeers',
    'DifficultCommunication',
    'DiversityImportant',
    'EnjoyDebugging',
    'ExCoder10Years',
    'ExCoderActive',
    'ExCoderBalance',
    'ExCoderBelonged',
    'ExCoderNotForMe',
    'ExCoderReturn',
    'ExCoderSkills',
    'ExCoderWillNotCode',
    'FriendsDevelopers',
    'InterestedAnswers',
    'InTheZone',
    'InvestTimeTools',
    'JobSecurity',
    'KinshipDevelopers',
    'LearningNewTech',
    'OtherPeoplesCode',
    'ProblemSolving',
    'ProjectManagement',
    'QuestionsConfusing',
    'QuestionsInteresting',
    'RightWrongWay',
    'SeriousWork',
    'ShipIt',
    'StackOverflowAdsDistracting',
    'StackOverflowAdsRelevant',
    'StackOverflowBetter',
    'StackOverflowCommunity',
    'StackOverflowHelpful',
    'StackOverflowMakeMoney',
    'StackOverflowModeration',
    'StackOverflowWhatDo',
    'SurveyLong',
    'UnderstandComputers',
    'WorkPayCare',
]

important_keys = [
    'AssessJobCommute',
    'AssessJobCompensation',
    'AssessJobDept',
    'AssessJobDiversity',
    'AssessJobExp',
    'AssessJobFinances',
    'AssessJobLeaders',
    'AssessJobOffice',
    'AssessJobProduct',
    'AssessJobProfDevel',
    'AssessJobProjects',
    'AssessJobRemote',
    'AssessJobRole',
    'AssessJobTech',
    'EducationImportant',
    'ImportantHiringAlgorithms',
    'ImportantHiringCommunication',
    'ImportantHiringCompanies',
    'ImportantHiringEducation',
    'ImportantHiringGettingThingsDone',
    'ImportantHiringOpenSource',
    'ImportantHiringPMExp',
    'ImportantHiringRep',
    'ImportantHiringTechExp',
    'ImportantHiringTitles',
]

satisfied_keys = [
    'CareerSat',
    'EquipmentSatisfiedCPU',
    'EquipmentSatisfiedMonitors',
    'EquipmentSatisfiedRAM',
    'EquipmentSatisfiedRW',
    'EquipmentSatisfiedStorage',
    'InfluenceInternet',
    'JobSat',
]

influence_keys = [
    'InfluenceWorkstation',
    'InfluenceHardware',
]

yes_no_keys = [
    'ClickyKeys',
    'Hobby',
    'Hobbyist',
    'OpenSource',
    'AdBlocker',
    'AdBlockerDisable',
    'StackOverflowHasAccount',
    'StackOverflowConsiderMember',
    'Dependents',
    'MilitaryUS',
    'MgrMoney',
    'FizzBuzz',
    'BetterLife',
    'OffOn',
]

last_three_months_keys = [
    'StackOverflowCopiedCode',
    'StackOverflowJobListing',
    'StackOverflowCompanyPage',
    'StackOverflowJobSearch',
    'StackOverflowNewQuestion',
    'StackOverflowAnswer',
    'StackOverflowMetaChat',
]

last_three_months_strs = {
    'several times': 5,
    'at least once each day': 4,
    'at least once each week': 3,
    'once or twice': 2,
    "haven't done at all": 1,
}

yes_no_strs = {
    'yes': 1,
    'no': 0,
    # todo replace "2" with nan/None
    "i'm not sure/i don't know": 2,
    "i'm not sure/i can't remember": 2,
    "i'm not sure": 2,
    "i'm not sure / i can't remember": 2,
    'not sure': 2,
    'what?': 2,
    "not sure / can't remember": 2
}

influence_strs = {
    'i am the final decision maker': 5,
    'a lot of influence': 4,
    'some influence': 3,
    'not much influence': 2,
    'no influence at all': 1,
}

agree_strs = {
    'strongly agree': 6,
    'agree': 5,
    'somewhat agree': 4,
    'neither agree nor disagree': 3,
    'somewhat disagree': 2,
    'disagree': 1,
    'strongly disagree': 0,
}

interested_strs = {
    'extremely interested': 4,
    'very interested': 3,
    'somewhat interested': 2,
    'a little bit interested': 1,
    'not at all interested': 0
}

satisfied_strs = {
    'very satisfied': 10,
    'extremely satisfied': 10,

    'moderately satisfied': 8.33,

    'satisfied': 6.66,
    'slightly satisfied': 6.66,

    'somewhat satisfied': 5,
    'neither satisfied nor dissatisfied': 5,

    'not very satisfied': 3.33,
    'slightly dissatisfied': 3.33,

    'moderately dissatisfied': 1.66,

    'not at all satisfied': 0,
    'very dissatisfied': 0,
    'extremely dissatisfied': 0
}

important_strs = {
    'very important': 5,
    'somewhat important': 4,
    'important': 3,
    'not very important': 2,
    'not at all important': 1,
}

overpaid_strs = {
    'greatly overpaid': 5,
    'somewhat overpaid': 4,
    'neither underpaid nor overpaid': 3,
    'somewhat underpaid': 2,
    'greatly underpaid': 1,
}

checkin_strs = {
    'never': 5,
    'just a few times over the year': 4,
    'less than once per month': 4,
    'a few times a month': 3,
    'weekly or a few times per month': 3,
    'a few times a week': 2,
    'a few times per week': 2,
    'once a day': 1,
    'multiple times a day': 0,
    'multiple times per day': 0
}

gender_strs = {
    'man': 1,
    'woman': 0,
    'male': 1,
    'female': 0
}

# to_drop = listvals

to_drop_before_start = [
    'AdBlocker',
    'AdBlockerDisable',
    'AdBlockerReasons',
    'AdsActions',
    'AdsAgreeDisagree1',
    'AdsAgreeDisagree2',
    'AdsAgreeDisagree3',
    'AdsPriorities1',
    'AdsPriorities2',
    'AdsPriorities3',
    'AdsPriorities4',
    'AdsPriorities5',
    'AdsPriorities6',
    'AdsPriorities7',
    'AIDangerous',
    'AIFuture',
    'AIInteresting',
    'AIResponsible',
    'BlockchainIs',
    'BlockchainOrg',
    'ClickyKeys',
    'CompFreq',
    'CompTotal',
    'Containers',
    'Currency',
    'CurrencyDesc',
    'CurrencySymbol',
    'EntTeams',
    'Ethnicity',
    'ExCoder10Years',
    'ExCoderActive',
    'ExCoderBalance',
    'ExCoderBelonged',
    'ExCoderNotForMe',
    'ExCoderReturn',
    'ExCoderSkills',
    'ExCoderWillNotCode',
    'FizzBuzz',
    'HackathonReasons',
    'HypotheticalTools1',
    'HypotheticalTools2',
    'HypotheticalTools3',
    'HypotheticalTools4',
    'HypotheticalTools5',
    'IDE',
    'InterestedAnswers',
    'ITperson',
    'JobProfile',
    'LastInt',
    'LearnedHiring',
    'MilitaryUS',
    'NonDeveloperType',
    'OffOn',
    'PronounceGIF',
    'QuestionsConfusing',
    'QuestionsInteresting',
    'RaceEthnicity',
    'Respondent',
    'ResumePrompted',
    'ResumeUpdate',
    'Salary',
    'SalaryType',
    'ScreenName',
    'SelfTaughtTypes',
    'Sexuality',
    'SexualOrientation',
    'SOAccount',
    'SocialMedia',
    'SOJobs',
    'SONewContent',
    'SOVisit1st',
    'StackOverflowAdsDistracting',
    'StackOverflowAdsRelevant',
    'StackOverflowDescribes',
    'StackOverflowDevices',
    'StackOverflowDevStory',
    'StackOverflowJobs',
    'StackOverflowJobsRecommend',
    'StackOverflowMakeMoney',
    'StackOverflowModeration',
    'SurveyEase',
    'SurveyEasy',
    'SurveyLength',
    'SurveyLong',
    'SurveyTooLong',
    'TabsSpaces',
    'TimeAfterBootcamp',
    'UpdateCV',
    'WelcomeChange',
    'YearsCodedJobPast',
]
to_drop = listvals + [
    # 'Gender',
]

replacers = [
    (influence_keys, influence_strs),
    (agree_keys, agree_strs),
    (satisfied_keys, satisfied_strs),
    (['CheckInCode'], checkin_strs),
    (['Overpaid'], overpaid_strs),
    (last_three_months_keys, last_three_months_strs),
    (important_keys, important_strs),
    (yes_no_keys, yes_no_strs),
    (['Gender'], gender_strs)
]

hdi = pd.read_csv(f'HDI.csv', index_col="Country")

failed_hdi = set()


def hdi_mapper_wrapper(year):
    def hdi_mapper(country):
        try:
            return hdi.at[country, f'{year}']
        except:
            failed_hdi.add(country)
            return None

    return hdi_mapper


def process(data, year):
    to_drop_first_filtered = list(set(data.columns).intersection(to_drop_before_start))
    data.drop(to_drop_first_filtered, axis=1, inplace=True)
    # data['gender_M'] = (data['Gender'] == 'Male').astype('int8')
    # data['gender_F'] = (data['Gender'] == 'Female').astype('int8')

    for keys, strs in replacers:
        cols = list(set(data.columns).intersection(keys))
        data[cols] = data[cols].applymap(dict_map(strs)).astype('float')

    listvals_filtered = list(set(data.columns).intersection(listvals))
    for index in listvals_filtered:
        data = pd.concat([data, userlist_to_cols(data[index])], axis=1)

    number_cols = list(set(data.columns).intersection(number_parses))
    data[number_cols] = data[number_cols].applymap(get_first_number).astype('float')

    data[['Country']] = data[['Country']].applymap(hdi_mapper_wrapper(year)).astype('str')

    to_drop_filtered = list(set(data.columns).intersection(to_drop))
    data.drop(to_drop_filtered, axis=1, inplace=True)
    # data = pd.get_dummies(data)
    # data.fillna(data.mean(), inplace=True)

    return data


def main():
    years = [2017, 2018, 2019]
    for year in years:
        data = pd.read_csv(f'data/{year}.csv')
        if year == 2019:
            data.rename(columns={"OpenSource": "OpenSourceQuality"}, inplace=True)
        elif year == 2018:
            data.rename(columns={"JobSatisfaction": "JobSat", "CareerSatisfaction": "CareerSat"}, inplace=True)
        # asdf = data['Country'].unique()
        uniqueValsOld = {}
        for x in data.columns:
            uniqueValsOld[x] = data[x].unique()

        data = process(data, year)
        # for x in data.columns:
        #     col = data[x].unique()

        uniqueVals = {}
        for x in data.columns:
            uniqueVals[x] = data[x].unique()
            # print(col)

        diff = {}
        for col in uniqueValsOld:
            oldLen = len(uniqueValsOld.get(col))
            newLen = 0
            if uniqueVals.get(col) is not None:
                newLen = len(uniqueVals.get(col))
            if oldLen != newLen and newLen != 0:
                diff[col] = (oldLen, newLen)

        types = data.select_dtypes(exclude=["float64", "int8"]).dtypes
        data.to_csv(f"data/{year}_pandas.csv", index=False)
    print()


if __name__ == '__main__':
    main()
