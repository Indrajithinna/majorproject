import { useState } from "react"


function Results() {

  const [usn, setUsn] = useState("")

  const [result, setResult] = useState([])

  const [loading, setLoading] = useState(false)

  const [error, setError] = useState("")


  async function searchResult() {

    const cleanUsn =
      usn.trim()


    if (!cleanUsn) {

      setError(
        "Please enter a valid USN."
      )

      setResult([])

      return

    }


    setLoading(true)

    setError("")


    try {

      const response =
        await fetch(
          `http://127.0.0.1:8001/result/${usn}`
        )


      if (!response.ok) {

        throw new Error(
          "Student result could not be retrieved."
        )

      }


      const data =
        await response.json()


      const rows =
        data.result || []


      if (rows.length === 0) {

        setError(
          "No result found for this USN."
        )

      }


      setResult(rows)

    }


    catch (err) {

      setResult([])

      setError(
        err.message ||
        "Unable to connect to the result service."
      )

    }


    finally {

      setLoading(false)

    }

  }


  return (

    <div className="page">


      <div className="page-heading">

        <h1>
          Student Results
        </h1>

        <p>
          Enter a university seat number
          to view the semester marksheet.
        </p>

      </div>



      <section className="panel search-panel">

        <div className="search-row">

          <input

            className="search-input"

            type="text"

            placeholder="Enter USN e.g. 1MG23IS001"

            value={usn}

            onChange={(e) =>
              setUsn(
                e.target.value.toUpperCase()
              )
            }

            onKeyDown={(e) =>
              e.key === "Enter" &&
              searchResult()
            }

          />


          <button
            className="btn btn-primary"
            onClick={searchResult}
            disabled={loading}
          >

            {loading
              ? "Searching..."
              : "View Result"
            }

          </button>

        </div>


        {error && (

          <div className="error">

            {error}

          </div>

        )}

      </section>



      {result.length > 0 && (

        <section className="result-card panel">


          <div className="student-header">

            <div>

              <h2 className="student-name">

                {result[0][0]}

              </h2>


              <div className="student-meta">

                USN:
                {" "}
                {result[0][1]}

                {" · "}

                Branch:
                {" "}
                {result[0][2]}

              </div>

            </div>


            <span className="badge">
              Official Result
            </span>

          </div>



          <div
            className="grid-3"
            style={{
              marginTop: "20px"
            }}
          >

            <div>

              <div className="stat-label">
                Student
              </div>

              <strong>
                {result[0][0]}
              </strong>

            </div>


            <div>

              <div className="stat-label">
                USN
              </div>

              <strong>
                {result[0][1]}
              </strong>

            </div>


            <div>

              <div className="stat-label">
                Branch
              </div>

              <strong>
                {result[0][2]}
              </strong>

            </div>

          </div>



          <div className="table-wrap">

            <table className="data-table">

              <thead>

                <tr>

                  <th>
                    Subject
                  </th>

                  <th>
                    Marks
                  </th>

                </tr>

              </thead>


              <tbody>

                {result.map(
                  (item, index) => (

                    <tr key={index}>

                      <td>
                        {item[4]}
                      </td>

                      <td>

                        <strong>
                          {item[5]}
                        </strong>

                      </td>

                    </tr>

                  )
                )}

              </tbody>

            </table>

          </div>


        </section>

      )}

    </div>

  )

}


export default Results