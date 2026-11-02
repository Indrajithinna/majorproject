import {
  useEffect,
  useState
} from "react"


function Branch() {

  const [data, setData] =
    useState([])

  const [loading, setLoading] =
    useState(true)

  const [error, setError] =
    useState("")


  async function loadData() {

    setLoading(true)

    setError("")


    try {

      const response =
        await fetch(
          "http://127.0.0.1:8001/branch-ranks"
        )


      if (!response.ok) {

        throw new Error(
          "Branch ranking service could not be reached."
        )

      }


      const result =
        await response.json()


      setData(
        result.branch_ranks || []
      )

    }


    catch (err) {

      setError(
        err.message ||
        "Unable to load branch rankings."
      )

    }


    finally {

      setLoading(false)

    }

  }


  useEffect(() => {

    loadData()

  }, [])


  return (

    <div className="page">


      <div className="page-heading">

        <h1>
          Branch Rankings
        </h1>

        <p>
          Academic performance ranking
          generated from university result data.
        </p>

      </div>



      <section className="panel">


        <div className="student-header">

          <div>

            <h2 style={{ margin: 0 }}>
              Student Performance Ranking
            </h2>

            <div className="student-meta">

              Higher average marks indicate
              stronger academic performance.

            </div>

          </div>


          <button
            className="btn btn-outline"
            onClick={loadData}
          >
            Refresh
          </button>

        </div>



        {loading && (

          <div className="loading">
            Loading rankings...
          </div>

        )}


        {error && (

          <div className="error">
            {error}
          </div>

        )}


        {!loading &&
          !error && (

            <div className="table-wrap">

              <table className="data-table">

                <thead>

                  <tr>

                    <th>
                      Rank
                    </th>

                    <th>
                      Student
                    </th>

                    <th>
                      Branch
                    </th>

                    <th>
                      Average
                    </th>

                  </tr>

                </thead>


                <tbody>

                  {data.map(
                    (item, index) => (

                      <tr key={index}>

                        <td>

                          <span className="rank">
                            {index + 1}
                          </span>

                        </td>


                        <td>

                          <strong>
                            {item[0]}
                          </strong>

                        </td>


                        <td>
                          {item[1]}
                        </td>


                        <td>

                          <strong>

                            {parseFloat(
                              item[2]
                            ).toFixed(2)}

                          </strong>

                        </td>

                      </tr>

                    )
                  )}

                </tbody>

              </table>

            </div>

          )}


        {!loading &&
          !error &&
          data.length === 0 && (

            <div className="loading">

              No ranking records are available.

            </div>

          )}

      </section>

    </div>

  )

}


export default Branch