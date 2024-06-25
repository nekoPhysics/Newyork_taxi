

def is_airport(df):
    # KJFK 40.639901, -73.806465 ~ 40.660645, -73.777591
    # KEWR 40.687314, -74.187967 ~ 40.697815, -74.176204
    df['is_airport'] = False
    df.loc[(df.pickup_latitude >= 40.639901) & (df.pickup_latitude <= 40.660645) & (df.pickup_longitude >= -73.806465) & (df.pickup_longitude <= -73.777591),'is_airport'] = True
    df.loc[(df.dropoff_latitude >= 40.639901) & (df.dropoff_latitude <= 40.660645) & (df.dropoff_longitude >= -73.806465) & (df.dropoff_longitude <= -73.777591),'is_airport'] = True
    df.loc[(df.pickup_latitude >= 40.687314) & (df.pickup_latitude <= 40.697815) & (df.pickup_longitude >= -74.187967) & (df.pickup_longitude <= -74.176204),'is_airport'] = True
    df.loc[(df.dropoff_latitude >= 40.687314) & (df.dropoff_latitude <= 40.697815) & (df.dropoff_longitude >= -74.187967) & (df.dropoff_longitude <= -74.176204),'is_airport'] = True
    return df


train_df = is_airport(train_df)


