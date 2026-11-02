import { useEffect, useState } from "react"

import { useNavigate } from "react-router-dom"



const PROMETHEUS = "/prometheus/api/v1/query"



function Analytics() {



  const navigate = useNavigate()



  const [metrics, setMetrics] = useState({

    actualCpu: null,

    predictedCpu: null,

    currentPods: null,

    recommendedPods: null,

    scalingScore: null,

    workload: "--",

    predictionActive: null,

    dataAge: null,

  })



  const [connected, setConnected] = useState(false)

  const [error, setError] = useState("")



  const queryPrometheus = async (query) => {



    const url =

      `${PROMETHEUS}?query=${encodeURIComponent(query)}`



    const response = await fetch(url)



    if (!response.ok) {

      throw new Error(

        `Prometheus HTTP ${response.status}`

      )

    }



    const data = await response.json()



    if (

      data.status !== "success" ||

      !data.data ||

      !data.data.result

    ) {

      return null

    }



    if (data.data.result.length === 0) {

      return null

    }



    return Number(

      data.data.result[0].value[1]

    )

  }





  const loadMetrics = async () => {



    try {



      const [
        actualCpu,
        predictedCpu,
        recommendedPods,
        scalingScore,
        predictionActive,
        dataAge,
        low,
        moderate,
        high,
        veryHigh,
        currentPods
      ] = await Promise.all([



        queryPrometheus(

          'workload_actual_cpu{deployment="mgrk-backend",namespace="default"}'

        ),



        queryPrometheus(

          'workload_predicted_cpu{deployment="mgrk-backend",namespace="default"}'

        ),



        queryPrometheus(

          'workload_recommended_pods{deployment="mgrk-backend",namespace="default"}'

        ),



        queryPrometheus(

          'workload_scaling_score{deployment="mgrk-backend",namespace="default"}'

        ),



        queryPrometheus(

          'workload_prediction_active{deployment="mgrk-backend",namespace="default"}'

        ),



        queryPrometheus(

          'workload_data_age_seconds{deployment="mgrk-backend",namespace="default"}'

        ),



        queryPrometheus(

          'workload_state{deployment="mgrk-backend",namespace="default",state="LOW"}'

        ),



        queryPrometheus(

          'workload_state{deployment="mgrk-backend",namespace="default",state="MODERATE"}'

        ),



        queryPrometheus(

          'workload_state{deployment="mgrk-backend",namespace="default",state="HIGH"}'

        ),



        queryPrometheus(

          'workload_state{deployment="mgrk-backend",namespace="default",state="VERY_HIGH"}'

        ),
        queryPrometheus(
          'workload_current_pods{deployment="mgrk-backend",namespace="default"}'
        )



      ])





      let workload = "--"



      if (veryHigh === 1) {

        workload = "VERY HIGH"

      }

      else if (high === 1) {

        workload = "HIGH"

      }

      else if (moderate === 1) {

        workload = "MODERATE"

      }

      else if (low === 1) {

        workload = "LOW"

      }





      const pods = currentPods





      setMetrics({



        actualCpu,



        predictedCpu,



        currentPods: pods,



        recommendedPods,



        scalingScore,



        workload,



        predictionActive,



        dataAge



      })



      setConnected(true)

      setError("")



    }

    catch (err) {



      console.error(

        "Analytics metrics error:",

        err

      )



      setConnected(false)



      setError(

        "Unable to connect to Prometheus"

      )



    }



  }





  useEffect(() => {



    loadMetrics()



    const interval = setInterval(

      loadMetrics,

      5000

    )



    return () =>

      clearInterval(interval)



  }, [])





  const formatNumber = (

    value,

    decimals = 1

  ) => {



    if (

      value === null ||

      value === undefined ||

      Number.isNaN(value)

    ) {

      return "--"

    }



    return Number(value).toFixed(decimals)



  }





  const cpuWidth = (value) => {



    if (

      value === null ||

      value === undefined

    ) {

      return "0%"

    }



    return `${Math.min(

      Math.max(value, 0),

      100

    )}%`



  }





  const workloadClass =

    metrics.workload

      .toLowerCase()

      .replace(" ", "-")





  return (



    <div className="analytics-page">



      <div className="analytics-container">





        {/* HEADER */}



        <div className="analytics-header">



          <h1>

            Academic & AI Analytics

          </h1>



          <p>

            Real-time system monitoring and proactive Kubernetes scaling.

          </p>



        </div>





        {/* LIVE STATUS */}



        <section>



          <h2>

            Live System Status

          </h2>



          <p className="section-description">

            Metrics automatically refresh every 5 seconds.

          </p>





          <div className="metric-grid">





            {/* ACTUAL CPU */}



            <div className="metric-card blue">



              <div className="metric-title">

                ACTUAL CPU

                <span>●</span>

              </div>



              <div className="metric-value">



                {formatNumber(

                  metrics.actualCpu

                )}

                %



              </div>



              <div className="metric-description">

                Live backend usage

              </div>



            </div>





            {/* PREDICTED CPU */}



            <div className="metric-card purple">



              <div className="metric-title">

                PREDICTED CPU

                <span>AI</span>

              </div>



              <div className="metric-value">



                {formatNumber(

                  metrics.predictedCpu

                )}

                %



              </div>



              <div className="metric-description">

                LSTM prediction

              </div>



            </div>





            {/* CURRENT PODS */}



            <div className="metric-card cyan">



              <div className="metric-title">

                CURRENT PODS

                <span>K8s</span>

              </div>



              <div className="metric-value">



                {metrics.currentPods ?? "--"}



              </div>



              <div className="metric-description">

                Active backend replicas

              </div>



            </div>





            {/* AI RECOMMENDED */}



            <div className="metric-card green">



              <div className="metric-title">

                AI RECOMMENDED

                <span>LSTM</span>

              </div>



              <div className="metric-value">



                {metrics.recommendedPods ?? "--"}



              </div>



              <div className="metric-description">

                Recommended replicas

              </div>



            </div>



          </div>





          {/* CONNECTION BAR */}



          <div className="connection-bar">



            <div>



              <span

                className={

                  connected

                    ? "connection-dot connected"

                    : "connection-dot"

                }

              />



              <strong>

                {connected

                  ? "Connected to Prometheus"

                  : "Connecting..."

                }

              </strong>



              {error && (

                <span className="connection-error">

                  {" "}— {error}

                </span>

              )}



            </div>



            <div>

              Auto refresh: 5 seconds

            </div>



          </div>



        </section>





        {/* CPU PREDICTION */}



        <section>



          <h2>

            CPU Prediction

          </h2>



          <p className="section-description">

            Actual CPU compared with the LSTM prediction.

          </p>





          <div className="two-column">





            {/* CPU UTILIZATION */}



            <div className="large-card">



              <h3>

                CPU Utilization

              </h3>





              <div className="progress-row">



                <div className="progress-header">



                  <span>

                    Actual CPU

                  </span>



                  <span>

                    {formatNumber(

                      metrics.actualCpu

                    )}

                    %

                  </span>



                </div>



                <div className="progress-background">



                  <div

                    className="progress-actual"

                    style={{

                      width:

                        cpuWidth(

                          metrics.actualCpu

                        )

                    }}

                  />



                </div>



              </div>





              <div className="progress-row">



                <div className="progress-header">



                  <span>

                    Predicted CPU

                  </span>



                  <span>

                    {formatNumber(

                      metrics.predictedCpu

                    )}

                    %

                  </span>



                </div>



                <div className="progress-background">



                  <div

                    className="progress-predicted"

                    style={{

                      width:

                        cpuWidth(

                          metrics.predictedCpu

                        )

                    }}

                  />



                </div>



              </div>



            </div>





            {/* KUBERNETES SCALING */}



            <div className="large-card">



              <h3>

                Kubernetes Scaling

              </h3>





              <div className="scaling-box">





                <div className="pod-box">



                  <span>

                    Current

                  </span>



                  <strong>

                    {metrics.currentPods ?? "--"}

                  </strong>



                  <small>

                    Pods

                  </small>



                </div>





                <div className="arrow">

                  →

                </div>





                <div className="pod-box recommendation">



                  <span>

                    AI Recommendation

                  </span>



                  <strong>

                    {metrics.recommendedPods ?? "--"}

                  </strong>



                  <small>

                    Pods

                  </small>



                </div>



              </div>



            </div>



          </div>



        </section>





        {/* PROACTIVE PIPELINE */}



        <section>



          <h2>

            Proactive Scaling Pipeline

          </h2>



          <p className="section-description">

            Live workload analysis and intelligent scaling decision.

          </p>





          <div className="pipeline">





            <div className="pipeline-step">



              <div className="pipeline-number">

                1

              </div>



              <strong>

                Monitoring

              </strong>



              <span>

                CPU & workload metrics

              </span>



            </div>





            <div className="pipeline-arrow">

              →

            </div>





            <div className="pipeline-step">



              <div className="pipeline-number">

                2

              </div>



              <strong>

                LSTM Prediction

              </strong>



              <span>

                Future CPU estimation

              </span>



            </div>





            <div className="pipeline-arrow">

              →

            </div>





            <div className="pipeline-step">



              <div className="pipeline-number">

                3

              </div>



              <strong>

                Workload Classifier

              </strong>



              <span className={`workload ${workloadClass}`}>

                {metrics.workload}

              </span>



            </div>





            <div className="pipeline-arrow">

              →

            </div>





            <div className="pipeline-step">



              <div className="pipeline-number">

                4

              </div>



              <strong>

                Kubernetes Scaling

              </strong>



              <span>

                {metrics.recommendedPods ?? "--"} pods

              </span>



            </div>



          </div>



        </section>





        {/* AI DETAILS */}



        <section>



          <h2>

            AI Scaling Decision

          </h2>





          <div className="info-grid">





            <div className="info-card">



              <span>

                Workload State

              </span>



              <strong>

                {metrics.workload}

              </strong>



            </div>





            <div className="info-card">



              <span>

                Scaling Score

              </span>



              <strong>

                {metrics.scalingScore === null

                  ? "--"

                  : `${(

                      metrics.scalingScore * 100

                    ).toFixed(1)}%`

                }

              </strong>



            </div>





            <div className="info-card">



              <span>

                Prediction Status

              </span>



              <strong>

                {metrics.predictionActive === 1

                  ? "ACTIVE"

                  : "INACTIVE"

                }

              </strong>



            </div>





            <div className="info-card">



              <span>

                Data Age

              </span>



              <strong>

                {metrics.dataAge === null

                  ? "--"

                  : `${metrics.dataAge.toFixed(1)}s`

                }

              </strong>



            </div>



          </div>



        </section>





        <button

          className="back-button"

          onClick={() => navigate("/")}

        >

          ← Back To Home

        </button>





      </div>





      <style>{`



        * {

          box-sizing: border-box;

        }



        .analytics-page {

          min-height: 100vh;

          background: #f5f8fc;

          color: #082b55;

          padding: 35px 20px 60px;

          font-family:

            Arial,

            Helvetica,

            sans-serif;

        }



        .analytics-container {

          max-width: 1180px;

          margin: auto;

        }



        .analytics-header {

          margin-bottom: 28px;

        }



        .analytics-header h1 {

          margin: 0;

          font-size: 38px;

          color: #07366b;

        }



        .analytics-header p {

          margin-top: 8px;

          color: #60789a;

          font-size: 16px;

        }



        section {

          margin-top: 25px;

        }



        section h2 {

          font-size: 25px;

          margin-bottom: 7px;

          color: #092d56;

        }



        .section-description {

          color: #58708e;

          margin-top: 0;

          margin-bottom: 17px;

        }



        .metric-grid {

          display: grid;

          grid-template-columns:

            repeat(4, 1fr);

          gap: 16px;

        }



        .metric-card {

          background: white;

          border-radius: 15px;

          padding: 24px;

          min-height: 145px;

          border-top: 3px solid;

          box-shadow:

            0 5px 20px

            rgba(25, 65, 105, 0.08);

        }



        .metric-card.blue {

          border-top-color: #2878ff;

        }



        .metric-card.purple {

          border-top-color: #7658ff;

        }



        .metric-card.cyan {

          border-top-color: #13a9b1;

        }



        .metric-card.green {

          border-top-color: #20a866;

        }



        .metric-title {

          display: flex;

          justify-content: space-between;

          font-size: 11px;

          font-weight: bold;

          color: #71849d;

          letter-spacing: .5px;

        }



        .metric-value {

          margin-top: 30px;

          font-size: 30px;

          font-weight: bold;

          color: #06366e;

        }



        .metric-description {

          margin-top: 12px;

          color: #71849d;

          font-size: 12px;

        }



        .connection-bar {

          background: #082849;

          color: white;

          padding: 13px 18px;

          border-radius: 10px;

          margin-top: 16px;

          display: flex;

          justify-content: space-between;

          align-items: center;

          font-size: 13px;

        }



        .connection-dot {

          display: inline-block;

          width: 9px;

          height: 9px;

          border-radius: 50%;

          background: #ffb000;

          margin-right: 8px;

        }



        .connection-dot.connected {

          background: #25d87b;

        }



        .connection-error {

          color: #ffb7b7;

        }



        .two-column {

          display: grid;

          grid-template-columns:

            1fr 1fr;

          gap: 20px;

        }



        .large-card {

          background: white;

          border-radius: 15px;

          padding: 25px;

          min-height: 240px;

          box-shadow:

            0 5px 20px

            rgba(25, 65, 105, 0.08);

        }



        .large-card h3 {

          font-size: 19px;

          margin-top: 0;

        }



        .progress-row {

          margin-top: 25px;

        }



        .progress-header {

          display: flex;

          justify-content: space-between;

          margin-bottom: 8px;

          font-size: 13px;

          color: #607897;

        }



        .progress-background {

          height: 10px;

          border-radius: 10px;

          background: #e8eef5;

          overflow: hidden;

        }



        .progress-actual {

          height: 100%;

          background: #2878ff;

          border-radius: 10px;

          transition: width .5s;

        }



        .progress-predicted {

          height: 100%;

          background: #7658ff;

          border-radius: 10px;

          transition: width .5s;

        }



        .scaling-box {

          display: flex;

          align-items: center;

          justify-content: center;

          gap: 20px;

          margin-top: 35px;

        }



        .pod-box {

          min-width: 130px;

          padding: 20px;

          border-radius: 14px;

          background: #f3f7fc;

          text-align: center;

          border: 1px solid #dce6f0;

        }



        .pod-box.recommendation {

          background: #effbf5;

          border-color: #bde8d0;

        }



        .pod-box span {

          display: block;

          font-size: 12px;

          color: #607897;

        }



        .pod-box strong {

          display: block;

          margin: 10px 0;

          font-size: 27px;

        }



        .pod-box small {

          color: #607897;

        }



        .arrow {

          font-size: 28px;

          color: #71849d;

        }



        .pipeline {

          background: white;

          padding: 25px;

          border-radius: 15px;

          box-shadow:

            0 5px 20px

            rgba(25, 65, 105, 0.08);

          display: flex;

          align-items: center;

          justify-content: space-between;

          gap: 10px;

        }



        .pipeline-step {

          flex: 1;

          text-align: center;

          padding: 15px;

        }



        .pipeline-number {

          width: 35px;

          height: 35px;

          line-height: 35px;

          margin: auto auto 10px;

          border-radius: 50%;

          background: #eaf2ff;

          color: #2878ff;

          font-weight: bold;

        }



        .pipeline-step strong {

          display: block;

          font-size: 14px;

        }



        .pipeline-step span {

          display: block;

          margin-top: 8px;

          color: #70839d;

          font-size: 12px;

        }



        .pipeline-arrow {

          font-size: 25px;

          color: #90a1b6;

        }



        .workload {

          font-weight: bold;

        }



        .workload.low {

          color: #15935a;

        }



        .workload.moderate {

          color: #c98200;

        }



        .workload.high,

        .workload.very-high {

          color: #d43c3c;

        }



        .info-grid {

          display: grid;

          grid-template-columns:

            repeat(4, 1fr);

          gap: 15px;

        }



        .info-card {

          background: white;

          border-radius: 13px;

          padding: 20px;

          box-shadow:

            0 5px 20px

            rgba(25, 65, 105, 0.07);

        }



        .info-card span {

          display: block;

          font-size: 12px;

          color: #70839d;

        }



        .info-card strong {

          display: block;

          margin-top: 9px;

          font-size: 20px;

          color: #08386d;

        }



        .back-button {

          margin-top: 30px;

          padding: 11px 20px;

          border: 0;

          border-radius: 8px;

          background: #1264d8;

          color: white;

          font-weight: bold;

          cursor: pointer;

        }



        @media (max-width: 900px) {



          .metric-grid {

            grid-template-columns:

              repeat(2, 1fr);

          }



          .two-column {

            grid-template-columns: 1fr;

          }



          .info-grid {

            grid-template-columns:

              repeat(2, 1fr);

          }



          .pipeline {

            flex-direction: column;

          }



          .pipeline-arrow {

            transform: rotate(90deg);

          }



        }



        @media (max-width: 600px) {



          .metric-grid,

          .info-grid {

            grid-template-columns: 1fr;

          }



          .analytics-header h1 {

            font-size: 29px;

          }



        }



      `}</style>



    </div>



  )



}



export default Analytics