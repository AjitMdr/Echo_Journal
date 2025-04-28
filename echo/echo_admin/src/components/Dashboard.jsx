import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { useAuth } from "../context/AuthContext";
import { ENDPOINTS } from "../config/api";

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884d8"];

const Dashboard = () => {
  const [metrics, setMetrics] = useState(null);
  const [trends, setTrends] = useState(null);
  const [subscriptionAnalytics, setSubscriptionAnalytics] = useState(null);
  const [userAnalytics, setUserAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { access_token } = useAuth();

  useEffect(() => {
    const fetchDashboardData = async () => {
      if (!access_token) {
        setError("No authentication token found");
        setLoading(false);
        return;
      }

      try {
        const headers = {
          Authorization: `Bearer ${access_token}`,
          "Content-Type": "application/json",
        };

        const responses = await Promise.all([
          fetch(ENDPOINTS.ADMIN.DASHBOARD.METRICS, { headers }),
          fetch(ENDPOINTS.ADMIN.DASHBOARD.TRENDS, { headers }),
          fetch(ENDPOINTS.ADMIN.DASHBOARD.SUBSCRIPTION_ANALYTICS, { headers }),
          fetch(ENDPOINTS.ADMIN.DASHBOARD.USER_ANALYTICS, { headers }),
        ]);

        const [metricsData, trendsData, subscriptionData, userData] =
          await Promise.all(responses.map((res) => res.json()));

        setMetrics(metricsData);
        setTrends(trendsData);
        setSubscriptionAnalytics(subscriptionData);
        setUserAnalytics(userData);
        setLoading(false);
      } catch (err) {
        console.error("Error fetching dashboard data:", err);
        setError("Failed to fetch dashboard data");
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [access_token]);

  if (loading) return <div className="p-4">Loading dashboard data...</div>;
  if (error) return <div className="p-4 text-red-600">{error}</div>;

  // Chart customization
  const chartConfig = {
    style: {
      background: "#ffffff",
      borderRadius: "8px",
      padding: "20px",
      boxShadow: "0 1px 3px rgba(0,0,0,0.12)",
    },
    grid: {
      strokeDasharray: "3 3",
      stroke: "#E5E7EB",
    },
    axis: {
      stroke: "#9CA3AF",
      fontSize: 12,
    },
    tooltip: {
      contentStyle: {
        background: "#ffffff",
        border: "none",
        borderRadius: "4px",
        boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
        padding: "8px",
      },
    },
  };

  return (
    <div className="p-6 space-y-6 bg-gray-50">
      {/* Overview Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Users"
          value={metrics?.overview?.total_users || 0}
          subtitle="All registered users"
        />
        <MetricCard
          title="Active Users"
          value={metrics?.overview?.active_users || 0}
          subtitle="Users active in last 30 days"
        />
        <MetricCard
          title="Premium Users"
          value={metrics?.overview?.premium_users || 0}
          subtitle="Current premium subscribers"
        />
        <MetricCard
          title="Monthly Revenue"
          value={`Rs. {
            metrics?.overview?.monthly_revenue?.toFixed(2) || "0.00"
          }`}
          subtitle="Current month revenue"
        />
      </div>

      {/* User Growth Trend */}
      <div className="bg-white p-6 rounded-lg shadow-sm">
        <h3 className="text-lg font-semibold mb-6">User Growth Trend</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={trends?.user_growth || []}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid {...chartConfig.grid} vertical={false} />
            <XAxis
              dataKey="trend_date"
              {...chartConfig.axis}
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <YAxis {...chartConfig.axis} />
            <Tooltip {...chartConfig.tooltip} />
            <Legend />
            <Line
              name="New Users"
              type="monotone"
              dataKey="count"
              stroke="#4F46E5"
              strokeWidth={2}
              dot={{ r: 4, fill: "#4F46E5" }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Journal Activity Trend */}
      <div className="bg-white p-6 rounded-lg shadow-sm">
        <h3 className="text-lg font-semibold mb-6">Journal Activity Trend</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={trends?.journal_activity || []}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid {...chartConfig.grid} vertical={false} />
            <XAxis
              dataKey="trend_date"
              {...chartConfig.axis}
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <YAxis {...chartConfig.axis} />
            <Tooltip {...chartConfig.tooltip} />
            <Legend />
            <Line
              name="Journal Entries"
              type="monotone"
              dataKey="count"
              stroke="#F97316"
              strokeWidth={2}
              dot={{ r: 4, fill: "#F97316" }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Additional Charts Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Revenue Trend */}
        <div className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-lg font-semibold mb-6">Revenue Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={trends?.revenue_trend || []}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid {...chartConfig.grid} vertical={false} />
              <XAxis
                dataKey="trend_date"
                {...chartConfig.axis}
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis {...chartConfig.axis} />
              <Tooltip {...chartConfig.tooltip} />
              <Legend />
              <Line
                name="Revenue"
                type="monotone"
                dataKey="revenue"
                stroke="#10B981"
                strokeWidth={2}
                dot={{ r: 4, fill: "#10B981" }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        
        {/* User Activity by Hour */}
        <div className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-lg font-semibold mb-6">User Activity by Hour</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={userAnalytics?.activity_by_hour || []}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid {...chartConfig.grid} vertical={false} />
              <XAxis dataKey="hour" {...chartConfig.axis} />
              <YAxis {...chartConfig.axis} />
              <Tooltip {...chartConfig.tooltip} />
              <Legend />
              <Line
                name="Activity"
                type="monotone"
                dataKey="count"
                stroke="#6366F1"
                strokeWidth={2}
                dot={{ r: 4, fill: "#6366F1" }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

const MetricCard = ({ title, value, subtitle }) => (
  <div className="bg-white p-6 rounded-lg shadow-sm">
    <h3 className="text-gray-600 text-sm font-medium">{title}</h3>
    <p className="text-3xl font-bold mt-2 text-gray-900">{value}</p>
    <p className="text-gray-500 text-sm mt-1">{subtitle}</p>
  </div>
);

export default Dashboard;
