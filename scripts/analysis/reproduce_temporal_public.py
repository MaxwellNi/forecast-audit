"""Freeze and run new temporal rating and past-price retail evaluations."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from temporal_public_forecasters import split_ratings, fit_rating_split, prepare_retail_tables, fit_retail_frame
from public_audit import audit_family, sha


def source_hashes():
    parent=Path(__file__).resolve().parent
    names=[Path(__file__).name,"temporal_public_forecasters.py","rating_forecasters.py",
        "retail_forecasters.py","public_audit.py","audit_panel_predictions.py",
        "cluster_covariance_reference.py"]
    return {name:sha(parent/name) for name in names}


def input_hashes(domain, source):
    paths=[source] if domain=="ratings" else [source/name for name in
        ("sales_train_evaluation.csv","calendar.csv","sell_prices.csv")]
    return [{"file":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in paths]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["freeze","run"])
    parser.add_argument("--domain",choices=["ratings","retail"],required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.stage=="freeze":
        args.output.mkdir(parents=True,exist_ok=False)
        protocol={"domain":args.domain,"source_hashes":source_hashes(),"inputs":input_hashes(args.domain,args.input),
            "recorded_utc":pd.Timestamp.now(tz="UTC").isoformat(),
            "record_type":"Local prospective analysis specification, not external preregistration",
            "rating_split":{"cutoff":"2017-01-01","end_exclusive":"2018-01-01","min_training_movie_count":200,
                "min_training_user_count":60,"max_training_eligible_users":6000,"seed":2026090503,
                "evaluation_activity_filter":False},
            "retail":{"test_complete_weeks":28,"training_eligible_series":4000,"seed":2026090504,
                "minimum_observed_training_price_weeks":8,"positive_training_sales_required":True,
                "target_week_price_allowed":False,"current_target_sales_allowed":False,
                "planned_calendar_covariates":["month_sin","month_cos"]},
            "model_family_size":10,"threads":4,
            "audit":{"qs":[8,12,16,24,32],"betas":[1,2],"primary_beta":1,
                "folds":5,"primary_fold_scheme":"contiguous","retail_hac_lag":2,"ratings_cluster_lag":0,
                "multiple_testing":"BY separately within complete ten-model family for each specification",
                "pvalue_status":"normal-reference diagnostic; dependence and nuisance bias not verified"},
            "model_parameters":"Ten documented algorithms unchanged except causal feature definition and frozen split; no tuning against evaluation outcomes"}
        (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
        print(json.dumps({"frozen_protocol_sha256":sha(args.output/"protocol.json")},indent=2))
        return
    protocol=json.loads((args.output/"protocol.json").read_text())
    if protocol["domain"]!=args.domain or protocol["source_hashes"]!=source_hashes():
        raise ValueError("domain or code changed after freeze")
    if protocol["inputs"]!=input_hashes(args.domain,args.input):
        raise ValueError("input changed after freeze")
    prediction_dir=args.output/"local_predictions";prediction_dir.mkdir(exist_ok=False)
    started=time.perf_counter()
    with threadpool_limits(limits=protocol["threads"]):
        if args.domain=="ratings":
            data=pd.read_csv(args.input)
            train,test,evidence=split_ratings(data)
            print(json.dumps({"stage":"prepared","domain":args.domain,**evidence}),flush=True)
            panel,forecasts=fit_rating_split(train,test)
            lag,frequency=0,None
        else:
            sales=pd.read_csv(args.input/"sales_train_evaluation.csv")
            calendar=pd.read_csv(args.input/"calendar.csv")
            prices=pd.read_csv(args.input/"sell_prices.csv")
            frame,cutoff,evidence=prepare_retail_tables(sales,calendar,prices)
            print(json.dumps({"stage":"prepared","domain":args.domain,**evidence}),flush=True)
            panel,forecasts=fit_retail_frame(frame,cutoff,threads=protocol["threads"])
            lag,frequency=2,"W-FRI"
        if len(forecasts)!=10:raise AssertionError("incomplete model family")
        errors=[];panels=[]
        for index,(name,prediction) in enumerate(forecasts.items()):
            prediction=np.asarray(prediction,dtype="<f8")
            if len(prediction)!=len(panel) or not np.isfinite(prediction).all():raise ValueError(name)
            frame=panel.assign(prediction=prediction,model=name)
            frame.to_parquet(prediction_dir/f"model_{index+1:02}.parquet",index=False)
            errors.append({"model":name,"n":len(frame),"mae":float(np.mean(np.abs(prediction-frame.y))),
                "rmse":float(np.sqrt(np.mean((prediction-frame.y)**2))),
                "prediction_sha256":hashlib.sha256(prediction.tobytes()).hexdigest()})
            panels.append((name,frame))
        print(json.dumps({"stage":"fitted","domain":args.domain,"models":len(forecasts)}),flush=True)
        extrapolated,single=audit_family(panels,lag,frequency)
    extrapolated.to_csv(args.output/"extrapolated_profile.csv",index=False)
    single.to_csv(args.output/"single_resolution_profile.csv",index=False)
    pd.DataFrame(errors).to_csv(args.output/"model_errors.csv",index=False)
    if protocol["source_hashes"]!=source_hashes():raise ValueError("source mutated during run")
    receipt={"domain":args.domain,"protocol_sha256":sha(args.output/"protocol.json"),"source_hashes":source_hashes(),
        "preparation":evidence,"elapsed_seconds":time.perf_counter()-started,"model_errors":errors,
        "dependencies":{n:importlib.metadata.version(n) for n in ("numpy","scipy","pandas","lightgbm","scikit-learn")},
        "all_ten_models_actually_refitted":True,"old_results_overwritten":False,
        "published_rows":False,"prospective_source_vintage_verified":False,
        "inference":"Prespecified temporal evaluation and nominal audit; no established universal calibration"}
    (args.output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"domain":args.domain,"complete":True,"elapsed_seconds":receipt["elapsed_seconds"]}),flush=True)


if __name__=="__main__":main()
