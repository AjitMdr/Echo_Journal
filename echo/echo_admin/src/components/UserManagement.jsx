import React, { useState, useEffect } from "react";
import api from "../utils/axios";
import {
  UserCircleIcon,
  ArrowPathIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  UsersIcon,
  UserGroupIcon,
  ChartBarIcon,
  PresentationChartLineIcon,
} from "@heroicons/react/24/outline";
import { ENDPOINTS } from "../config/api";

const UserManagement = () => {
  const [users, setUsers] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [filter, setFilter] = useState("all");
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [usersResponse, analyticsResponse] = await Promise.all([
        api.get("/api/admin/dashboard/user_management/"),
        api.get("/api/admin/dashboard/user_analytics/"),
      ]);

      const nonAdminUsers = usersResponse.data.users.filter(
        (user) => !user.is_staff
      );
      setUsers(nonAdminUsers);
      setAnalytics(analyticsResponse.data);
    } catch (err) {
      setError("Failed to fetch user data. Please try again.");
      console.error("Error fetching user data:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleActive = async (userId) => {
    try {
      await api.post(`/api/admin/users/${userId}/toggle_active/`);
      fetchData();
    } catch (err) {
      setError("Failed to toggle user status. Please try again.");
      console.error("Error toggling user status:", err);
    }
  };

  const filteredUsers = users.filter((user) => {
    const matchesSearch =
      user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.email.toLowerCase().includes(searchTerm.toLowerCase());

    switch (filter) {
      case "active":
        return matchesSearch && user.is_active;
      case "inactive":
        return matchesSearch && !user.is_active;
      default:
        return matchesSearch;
    }
  });

  const MetricCard = ({ title, value, subtitle, icon: Icon, color }) => (
    <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="mt-2 text-3xl font-bold text-gray-900">{value}</p>
          <p className="mt-1 text-sm text-gray-500">{subtitle}</p>
        </div>
        <div className={`p-3 rounded-full bg-${color}-100`}>
          <Icon className={`w-6 h-6 text-${color}-600`} />
        </div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <ArrowPathIcon className="w-8 h-8 animate-spin text-blue-500" />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {error}
        </div>
      )}

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Total Users"
          value={analytics?.retention_metrics?.total_users || 0}
          subtitle="Total registered users"
          icon={UsersIcon}
          color="blue"
        />
        <MetricCard
          title="Active Users"
          value={analytics?.retention_metrics?.active_users || 0}
          subtitle="Users active in last 30 days"
          icon={UserGroupIcon}
          color="green"
        />
        {/* <MetricCard
          title="Retention Rate"
          value={`${analytics?.retention_metrics?.login_retention_rate || 0}%`}
          subtitle="User retention rate"
          icon={ChartBarIcon}
          color="purple"
        /> */}
        <MetricCard
          title="Engagement Rate"
          value={`${analytics?.retention_metrics?.engagement_rate || 0}%`}
          subtitle="User engagement rate"
          icon={PresentationChartLineIcon}
          color="indigo"
        />
      </div>

      {/* Search and Filter Bar */}
      <div className="mb-6 flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search users..."
            className="pl-10 pr-4 py-2 w-full border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="flex items-center gap-2">
          <FunnelIcon className="h-5 w-5 text-gray-400" />
          <select
            className="border border-gray-200 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            <option value="all">All Users</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
          </select>
        </div>
      </div>

      {/* Users Table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                User
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Joined Date
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {filteredUsers.map((user) => (
              <tr key={user.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <div className="flex-shrink-0 h-10 w-10">
                      {user.profile_picture ? (
                        <img
                          src={user.profile_picture}
                          alt={`${user.username}'s profile`}
                          className="h-10 w-10 rounded-full object-cover"
                          onError={(e) => {
                            e.target.onerror = null;
                            e.target.src = null;
                            e.target.className = "hidden";
                            e.target.nextSibling.className =
                              "h-10 w-10 text-gray-300 rounded-full";
                          }}
                        />
                      ) : (
                        <UserCircleIcon className="h-10 w-10 text-gray-300 rounded-full" />
                      )}
                    </div>
                    <div className="ml-4">
                      <div className="text-sm font-medium text-gray-900">
                        {user.username}
                      </div>
                      <div className="text-sm text-gray-500">{user.email}</div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      user.is_active
                        ? "bg-green-100 text-green-800"
                        : "bg-red-100 text-red-800"
                    }`}
                  >
                    {user.is_active ? "Active" : "Inactive"}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {new Date(user.date_joined).toLocaleDateString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  <button
                    onClick={() => handleToggleActive(user.id)}
                    className={`inline-flex items-center px-3 py-1.5 border rounded-lg text-sm font-medium ${
                      user.is_active
                        ? "border-red-200 text-red-600 hover:bg-red-50"
                        : "border-green-200 text-green-600 hover:bg-green-50"
                    } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
                  >
                    {user.is_active ? "Deactivate" : "Activate"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default UserManagement;
