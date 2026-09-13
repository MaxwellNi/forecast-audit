"""Independent time-isolation checks for the new public forecasting tasks."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/analysis"))
from temporal_public_forecasters import split_ratings, fit_rating_split, prepare_retail_tables, fit_retail_frame


def ratings_fixture():
    rng=np.random.default_rng(9417)
    rows=[]
    for user in range(70):
        for movie in range(60):
            stamp=pd.Timestamp("2017-03-01",tz="UTC") if movie%4==user%4 else pd.Timestamp("2016-06-01",tz="UTC")
            rows.append((user,movie,float(rng.integers(1,11)/2),int(stamp.timestamp())))
    return pd.DataFrame(rows,columns=["userId","movieId","rating","timestamp"])


def retail_fixture():
    n_weeks=20; n_series=5
    dates=pd.date_range("2011-01-29",periods=7*n_weeks)
    calendar=pd.DataFrame({"date":dates.strftime("%Y-%m-%d"),"d":[f"d_{i+1}" for i in range(len(dates))],
        "wm_yr_wk":np.repeat(np.arange(100,100+n_weeks),7)})
    rng=np.random.default_rng(2043)
    sales=pd.DataFrame({"store_id":"S", "item_id":[f"I{i}" for i in range(n_series)]})
    values=pd.DataFrame(rng.poisson(3,size=(n_series,len(dates))),columns=calendar.d)
    sales=pd.concat([sales,values],axis=1)
    prices=pd.DataFrame([(f"I{i}","S",w,1+i*.1+(w%3)*.01) for i in range(n_series) for w in range(100,120)],
        columns=["item_id","store_id","wm_yr_wk","sell_price"])
    return sales,calendar,prices


class TemporalPublicTests(unittest.TestCase):
    def test_rating_cohort_never_uses_evaluation_activity_or_values(self):
        frame=ratings_fixture()
        train,test,evidence=split_ratings(frame,min_user=10,min_movie=10,max_users=50)
        changed=frame.copy(); future=changed.timestamp>=int(pd.Timestamp("2017-01-01",tz="UTC").timestamp())
        changed.loc[future,"rating"] *= -50
        # A very active new future user cannot enter the training cohort.
        extra=changed.loc[future].copy();extra.userId += 1000
        second,_,_=split_ratings(pd.concat([changed,extra],ignore_index=True),min_user=10,min_movie=10,max_users=50)
        pd.testing.assert_frame_equal(train,second)
        self.assertLess(train.timestamp.max(),test.timestamp.min())
        self.assertFalse(evidence["cohort_uses_evaluation_rows"])

    def test_all_ten_rating_forecasts_ignore_evaluation_targets(self):
        frame=ratings_fixture()
        train,test,_=split_ratings(frame,min_user=10,min_movie=10,max_users=60)
        _,first=fit_rating_split(train,test)
        changed=test.copy();changed.rating=np.arange(len(changed))*100
        _,second=fit_rating_split(train,changed)
        self.assertEqual(len(first),10)
        for name in first:np.testing.assert_allclose(first[name],second[name],rtol=0,atol=1e-10)

    def test_current_and_future_retail_prices_do_not_change_current_features(self):
        sales,calendar,prices=retail_fixture()
        first,cutoff,_=prepare_retail_tables(sales,calendar,prices,n_series=5,test_weeks=4)
        changed=prices.copy();changed.loc[changed.wm_yr_wk>=cutoff,"sell_price"] *= 100
        second,_,_=prepare_retail_tables(sales,calendar,changed,n_series=5,test_weeks=4)
        columns=["series","wm_yr_wk","price_lag1_rel","price_change_lag1","month_sin","month_cos"]
        pd.testing.assert_frame_equal(first.loc[first.wm_yr_wk<=cutoff,columns],second.loc[second.wm_yr_wk<=cutoff,columns])

    def test_all_ten_retail_forecasts_ignore_current_and_future_sales(self):
        sales,calendar,prices=retail_fixture()
        first,cutoff,evidence=prepare_retail_tables(sales,calendar,prices,n_series=5,test_weeks=4)
        changed=sales.copy()
        future_days=calendar.loc[calendar.wm_yr_wk>=cutoff,"d"].tolist()
        changed[future_days]=changed[future_days]*100+73
        second,_,_=prepare_retail_tables(changed,calendar,prices,n_series=5,test_weeks=4)
        panel1,pred1=fit_retail_frame(first,cutoff,threads=1)
        panel2,pred2=fit_retail_frame(second,cutoff,threads=1)
        current=panel1.period==panel1.period.min()
        self.assertEqual(len(pred1),10)
        for name in pred1:np.testing.assert_allclose(pred1[name][current],pred2[name][current],rtol=0,atol=1e-10)
        self.assertEqual(evidence["test_rows"],20)

    def test_missing_evaluation_price_does_not_select_rows(self):
        sales,calendar,prices=retail_fixture()
        first,cutoff,_=prepare_retail_tables(sales,calendar,prices,n_series=5,test_weeks=4)
        prices.loc[prices.wm_yr_wk>=cutoff,"sell_price"]=np.nan
        second,_,_=prepare_retail_tables(sales,calendar,prices,n_series=5,test_weeks=4)
        pd.testing.assert_frame_equal(first[["series","wm_yr_wk","sales"]],second[["series","wm_yr_wk","sales"]])
        self.assertTrue(np.isfinite(second[["price_lag1_rel","price_change_lag1"]]).all().all())


if __name__=="__main__":unittest.main()
