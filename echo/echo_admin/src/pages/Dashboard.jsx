import React, { useState, useEffect } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE_URL = "http://100.64.193.236:8000";

// Add print styles
const printStyles = `
  @media print {
    @page {
      size: A4;
      margin: 20mm;
    }
    
    body {
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
    }

    /* Hide non-dashboard elements */
    nav, 
    header,
    footer,
    .sidebar,
    .navbar,
    .login-form,
    .auth-container,
    .no-print,
    button {
      display: none !important;
    }

    /* Ensure dashboard content takes full width */
    .dashboard-container {
      width: 100% !important;
      margin: 0 !important;
      padding: 0 !important;
    }

    /* Optimize chart display */
    canvas {
      max-width: 100% !important;
      height: auto !important;
    }

    /* Add page breaks where needed */
    .page-break {
      page-break-before: always;
    }

    /* Enhance text readability for print */
    .text-gray-500,
    .text-gray-600,
    .text-gray-700 {
      color: #000 !important;
    }

    /* Ensure white backgrounds for cards */
    .bg-white {
      background-color: #fff !important;
      box-shadow: none !important;
    }
  }
`;

const Dashboard = () => {
  const [metrics, setMetrics] = useState(null);
  const [trends, setTrends] = useState(null);
  const [subscriptionAnalytics, setSubscriptionAnalytics] = useState(null);
  const [userAnalytics, setUserAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        // Get token from localStorage
        const token = localStorage.getItem("access_token");
        if (!token) {
          throw new Error("No authentication token found");
        }

        // Configure axios headers
        const config = {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        };

        const [metricsRes, trendsRes, subscriptionRes, userRes] =
          await Promise.all([
            axios.get(`${API_BASE_URL}/api/admin/dashboard/metrics/`, config),
            axios.get(`${API_BASE_URL}/api/admin/dashboard/trends/`, config),
            axios.get(
              `${API_BASE_URL}/api/admin/dashboard/subscription_analytics/`,
              config
            ),
            axios.get(
              `${API_BASE_URL}/api/admin/dashboard/user_analytics/`,
              config
            ),
          ]);

        console.log("Metrics Response:", metricsRes.data);
        console.log("Trends Response:", trendsRes.data);
        console.log("Subscription Response:", subscriptionRes.data);
        console.log("User Analytics Response:", userRes.data);

        setMetrics(metricsRes.data.overview);
        setTrends(trendsRes.data);
        setSubscriptionAnalytics(subscriptionRes.data);
        setUserAnalytics(userRes.data);
        setLoading(false);
      } catch (err) {
        console.error("Dashboard data fetch error:", err);
        setError(err.message);
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const handlePrintReport = () => {
    window.print();
  };

  if (loading)
    return (
      <div className="flex justify-center items-center h-screen">
        Loading...
      </div>
    );
  if (error) return <div className="text-red-500 p-4">Error: {error}</div>;

  const formatCurrency = (value) => {
    return `Rs. ${(value || 0).toFixed(2)}`;
  };

  const formatPercentage = (value) => {
    return `${(value || 0).toFixed(2)}%`;
  };

  // Chart data preparation
  const userGrowthData = {
    labels:
      trends?.user_growth?.map((item) =>
        new Date(item.trend_date).toLocaleDateString()
      ) || [],
    datasets: [
      {
        label: "New Users",
        data: trends?.user_growth?.map((item) => item.count) || [],
        borderColor: "rgb(59, 130, 246)",
        tension: 0.1,
        fill: false,
      },
    ],
  };

  const journalActivityData = {
    labels:
      trends?.journal_activity?.map((item) =>
        new Date(item.trend_date).toLocaleDateString()
      ) || [],
    datasets: [
      {
        label: "Journal Entries",
        data: trends?.journal_activity?.map((item) => item.count) || [],
        borderColor: "rgb(249, 115, 22)",
        tension: 0.1,
        fill: false,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <div className="dashboard-container p-6">
      <style>{printStyles}</style>
      {/* Print Report Button */}
      <div className="flex justify-end mb-6 no-print">
        <button
          onClick={handlePrintReport}
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg shadow-sm flex items-center"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5 mr-2"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z"
            />
          </svg>
          Print Report
        </button>
      </div>

      {/* Report Title - Only shows in print */}
      <div className="hidden print:block mb-8">
        <h1 className="text-3xl font-bold text-center">
          Echo Journal Dashboard Report
        </h1>
        <p className="text-center text-gray-600 mt-2">
          Generated on {new Date().toLocaleDateString()}
        </p>
        <hr className="my-4 border-gray-300" />
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-700">Total Users</h3>
          <p className="text-3xl font-bold text-blue-600">
            {metrics?.total_users || 0}
          </p>
          <p className="text-sm text-gray-500">
            Active: {metrics?.active_users || 0}
          </p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-700">Revenue</h3>
          <p className="text-3xl font-bold text-green-600">
            {formatCurrency(metrics?.total_revenue)}
          </p>
          <p className="text-sm text-gray-500">
            Monthly: {formatCurrency(metrics?.monthly_revenue)}
          </p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-700">Subscriptions</h3>
          <p className="text-3xl font-bold text-purple-600">
            {metrics?.active_subscriptions || 0}
          </p>
          <p className="text-sm text-gray-500">
            Premium: {metrics?.premium_users || 0}
          </p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-700">Journals</h3>
          <p className="text-3xl font-bold text-orange-600">
            {metrics?.total_journals || 0}
          </p>
          <p className="text-sm text-gray-500">
            Today: {metrics?.today_journals || 0}
          </p>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* User Growth Trend */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">User Growth Trend</h3>
          {trends?.user_growth?.length > 0 ? (
            <Line data={userGrowthData} options={chartOptions} />
          ) : (
            <div className="text-gray-500 text-center py-8">
              No data available
            </div>
          )}
        </div>

        {/* Journal Activity Trend */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Journal Activity Trend</h3>
          {trends?.journal_activity?.length > 0 ? (
            <Line data={journalActivityData} options={chartOptions} />
          ) : (
            <div className="text-gray-500 text-center py-8">
              No data available
            </div>
          )}
        </div>
      </div>

      {/* Analytics Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Subscription Analytics */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Subscription Analytics</h3>
          <div className="space-y-4">
            {subscriptionAnalytics?.plan_distribution?.map((plan, index) => (
              <div key={index} className="flex justify-between items-center">
                <span className="text-gray-600">{plan.plan__name}</span>
                <span className="font-semibold">{plan.count} users</span>
              </div>
            ))}
          </div>
        </div>

        {/* User Analytics */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">User Analytics</h3>
          <div className="space-y-4">
            <div>
              <p className="text-sm text-gray-500">Average Journals per User</p>
              <p className="text-xl font-bold text-orange-600">
                {(metrics?.avg_journals_per_user || 0).toFixed(2)}
              </p>
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>
          <div className="space-y-4">
            <div>
              <p className="text-sm text-gray-500">Active Users Ratio</p>
              <p className="text-xl font-bold text-purple-600">
                {formatPercentage(metrics?.active_users_ratio)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Premium Conversion Rate</p>
              <p className="text-xl font-bold text-green-600">
                {formatPercentage(metrics?.premium_conversion_rate)}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
