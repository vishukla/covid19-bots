def get_botometer_score(handle):
    '''
    This function takes in a Twitter handle in string format and returns universal display score
    assigned by Botometer API. For more info refer https://botometer.iuni.iu.edu/.
    The consumer and access keys and tokens are for Twitter account and rapidapi key is issued
    by RapidAPI for Botometer API consumption.
    '''
    import botometer
    try:
        twitter_app_auth = {'consumer_key': 'xxxxxxxxxxxxxxxxxxxxxxxxxxx',\
                            'consumer_secret': 'xxxxxxxxxxxxxxxxxxxxxxxxxxx',\
                            'access_token': 'xxxxxxxxxxxxxxxxxxxxxxxxxxx',\
                            'access_token_secret': 'xxxxxxxxxxxxxxxxxxxxxxxxxxx'}
        rapidapi_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxx'
        bom = botometer.Botometer(wait_on_ratelimit=True,
                              rapidapi_key=rapidapi_key,
                              **twitter_app_auth)
        result = bom.check_account(str('@' + handle))
        return result['display_scores']['universal']
    except Exception as e:
        print(str(e))
        return None