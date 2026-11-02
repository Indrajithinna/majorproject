import {
  BrowserRouter,
  Routes,
  Route
} from "react-router-dom"

import Home from "./pages/Home"
import Results from "./pages/Results"
import Branch from "./pages/Branch"
import Analytics from "./pages/Analytics"


function App() {

  return (

    <BrowserRouter>

      <Routes>

        <Route
          path="/"
          element={<Home />}
        />

        <Route
          path="/results"
          element={<Results />}
        />

        <Route
          path="/branch"
          element={<Branch />}
        />

        <Route
          path="/analytics"
          element={<Analytics />}
        />

      </Routes>

    </BrowserRouter>

  )

}


export default App