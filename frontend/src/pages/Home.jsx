import { useNavigate } from "react-router-dom"


function Home() {

  const navigate = useNavigate()


  return (

    <div className="landing-page">

      <div className="landing-background" />

      <div className="landing-overlay" />


      <div className="landing-frame">


        {/* NAVBAR */}

        <header className="landing-nav">

          <div className="landing-logo">

            <div className="logo-symbol">
              M
            </div>

            <span>
              MGRK University
            </span>

          </div>


          <nav className="landing-menu">

            <button
              onClick={() => navigate("/")}
            >
              Home
            </button>


            <button
              onClick={() => navigate("/results")}
            >
              Results
            </button>


            <button
              onClick={() => navigate("/branch")}
            >
              Rankings
            </button>


            <button
              onClick={() => navigate("/analytics")}
            >
              Analytics
            </button>


            <button
              className="nav-book-button"
              onClick={() => navigate("/results")}
            >
              Check Result
            </button>

          </nav>

        </header>



        {/* HERO */}

        <section className="landing-hero">

          <div className="hero-heading">

            <div>
              Academic
            </div>

            <div className="hero-faded">
              Excellence
            </div>

            <div>
              Redefined
            </div>

          </div>


          <p className="hero-description">

            Access student results, academic
            rankings and intelligent performance
            analytics through the MGRK University
            portal.

          </p>

        </section>



        {/* BOTTOM INFORMATION */}

        <div className="landing-info">

          <div className="info-line" />

          <p>

            A modern academic information
            platform designed for students,
            faculty and university administration.

          </p>

        </div>



        {/* GLASS CARD */}

        <div className="result-glass-card">

          <div className="card-top">

            <div>

              <small>
                MGRK UNIVERSITY
              </small>

              <h3>
                Student Portal
              </h3>

            </div>


            <span className="card-icon">
              ↗
            </span>

          </div>


          <div className="card-details">

            <div>

              <span>
                Students
              </span>

              <strong>
                2000+
              </strong>

            </div>


            <div>

              <span>
                Records
              </span>

              <strong>
                10K+
              </strong>

            </div>


            <div>

              <span>
                Status
              </span>

              <strong className="online">
                ● Online
              </strong>

            </div>

          </div>


          <div className="card-actions">

            <button
              onClick={() => navigate("/results")}
            >
              View Student Result
            </button>

          </div>

        </div>



        {/* AI LABEL */}

        <div className="landing-rating">

          <strong>
            AI
          </strong>

          <span>
            LSTM
          </span>

          <small>
            Proactive scaling
          </small>

        </div>



        {/* SCROLL */}

        <div className="scroll-indicator">

          <span />

          Explore portal

        </div>


      </div>

    </div>

  )

}


export default Home