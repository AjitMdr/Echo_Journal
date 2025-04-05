import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:echo_fe/services/auth/secure_storage_service.dart';
import 'package:echo_fe/core/configs/api_config.dart';

class FriendsService {
  final Dio _dio = Dio();

  FriendsService() {
    _dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) async {
          // Add auth token to every request
          final token = await SecureStorageService.getAccessToken();
          if (token != null) {
            options.headers['Authorization'] = 'Bearer $token';
            print('🔒 Added auth token to request: ${options.uri}');
          } else {
            print('❌ No auth token available for request: ${options.uri}');
          }
          return handler.next(options);
        },
        onError: (DioException e, handler) {
          print('🚨 DioError: ${e.type} - ${e.message}');
          if (e.response != null) {
            print(
              '🚨 Response: ${e.response?.statusCode} - ${e.response?.data}',
            );
          }
          return handler.next(e);
        },
      ),
    );
  }

  Future<String?> _getToken() async {
    return await SecureStorageService.getAccessToken();
  }

  // Get all friends
  Future<List<Map<String, dynamic>>> getFriends() async {
    final accessToken = await _getToken();
    if (accessToken == null) throw Exception('User is not authenticated');

    try {
      debugPrint('🔄 Getting friends list...');
      final response =
          await _dio.get(ApiConfig.getFullUrl('friends/friendships'));
      debugPrint('📡 Friends Response: ${response.data}');

      if (response.statusCode == 200) {
        if (response.data is List) {
          final List<dynamic> friendships = response.data;
          debugPrint('✅ Received ${friendships.length} friendships');

          // Get current user ID
          String? currentUserId = await SecureStorageService.getUserId();
          debugPrint('👤 Current user ID: ${currentUserId ?? 'not found'}');

          if (currentUserId == null || currentUserId.isEmpty) {
            throw Exception('Could not determine current user ID');
          }

          // Transform the data into the format expected by the UI
          final List<Map<String, dynamic>> processedFriends = [];

          for (var friendship in friendships) {
            // Validate friendship data
            if (friendship['user1'] == null ||
                friendship['user2'] == null ||
                friendship['user1']['id'] == null ||
                friendship['user2']['id'] == null) {
              debugPrint('⚠️ Skipping invalid friendship data');
              continue;
            }

            // Convert IDs to strings for comparison
            final user1Id = friendship['user1']['id'].toString();
            final user2Id = friendship['user2']['id'].toString();

            // Skip if this is a self-friendship
            if (user1Id == currentUserId && user2Id == currentUserId) {
              debugPrint('⚠️ Skipping self-friendship');
              continue;
            }

            // Get the other user's data (not the current user)
            Map<String, dynamic>? otherUser;
            String? friendshipId = friendship['id']?.toString();

            if (user1Id == currentUserId) {
              otherUser = friendship['user2'];
              debugPrint('👥 Found friend in user2: ${otherUser?['username']}');
            } else if (user2Id == currentUserId) {
              otherUser = friendship['user1'];
              debugPrint('👥 Found friend in user1: ${otherUser?['username']}');
            } else {
              debugPrint(
                  '⚠️ Skipping friendship - neither user matches current user');
              continue;
            }

            // Skip if otherUser is null or missing required data
            if (otherUser == null ||
                otherUser['id'] == null ||
                otherUser['username'] == null) {
              debugPrint('⚠️ Skipping friendship - invalid other user data');
              continue;
            }

            // Create friend data from the other user
            final friendData = {
              'id': otherUser['id'].toString(),
              'name': otherUser['username'] as String,
              'username': otherUser['username'] as String,
              'email': otherUser['email'] as String? ?? '',
              'friendship_id': friendshipId ?? '',
              'created_at': friendship['created_at'] as String? ?? '',
              'current_streak': otherUser['current_streak'] as int? ?? 0,
              'longest_streak': otherUser['longest_streak'] as int? ?? 0,
            };

            // Add the friend to the list
            processedFriends.add(friendData);
            debugPrint(
                '👥 Added friend: ${friendData['username']} (ID: ${friendData['id']})');
          }

          debugPrint('✅ Processed ${processedFriends.length} friends');
          return processedFriends;
        } else {
          debugPrint('❌ Unexpected response format: ${response.data}');
          throw Exception('Unexpected response format from server');
        }
      }
      throw Exception(response.data['message'] ?? 'Failed to fetch friends');
    } on DioException catch (e) {
      debugPrint('❌ Error fetching friends: ${e.message}');
      if (e.response?.data != null) {
        debugPrint('❌ Error response: ${e.response?.data}');
        throw Exception(
            e.response?.data['message'] ?? 'Network error occurred');
      }
      throw Exception('Network error occurred');
    }
  }

  // Search for users
  Future<Map<String, dynamic>> searchUsers(String query, {int page = 1}) async {
    final accessToken = await _getToken();
    if (accessToken == null) throw Exception('User is not authenticated');

    try {
      print('🔍 Searching users with query: $query, page: $page'); // Debug log
      print('🔑 Using token: $accessToken'); // Debug log

      final response = await _dio.get(
        ApiConfig.getFullUrl('friends/search/'),
        queryParameters: {'search': query, 'page': page, 'page_size': 10},
      );

      print('📡 Search Response Status: ${response.statusCode}'); // Debug log
      print('📡 Search Response Data: ${response.data}'); // Debug log

      if (response.statusCode == 200) {
        final results = {
          'results': (response.data['results'] as List?)?.map((user) {
                final userData = Map<String, dynamic>.from(user);
                print(
                  '👤 User ${userData['username']} - Status: ${userData['friendship_status']}',
                ); // Debug log
                return userData;
              }).toList() ??
              [],
          'next': response.data['next'],
          'previous': response.data['previous'],
          'count': response.data['count'],
        };
        print('✅ Processed results: $results'); // Debug log
        return results;
      }
      print('❌ Error response: ${response.data}'); // Debug log
      throw Exception(response.data['message'] ?? 'Failed to search users');
    } on DioException catch (e) {
      print('❌ DioException: ${e.message}'); // Debug log
      print('❌ DioException response: ${e.response?.data}'); // Debug log
      throw Exception(e.response?.data?['message'] ?? 'Network error occurred');
    }
  }

  // Send friend request
  Future<void> sendFriendRequest(dynamic toUserId) async {
    final accessToken = await _getToken();
    if (accessToken == null) throw Exception('User is not authenticated');

    try {
      // Convert toUserId to int if it's a string
      final userId = toUserId is String ? int.parse(toUserId) : toUserId;

      print('📤 Sending friend request to user ID: $userId'); // Debug log
      print('📤 Request data: {"to_user_id": $userId}'); // Debug log

      final response = await _dio.post(
        ApiConfig.getFullUrl('friends/requests/'),
        data: {'to_user_id': userId},
      );

      print('📡 Response status: ${response.statusCode}'); // Debug log
      print('📡 Response data: ${response.data}'); // Debug log

      if (response.statusCode != 201) {
        String errorMessage = 'Failed to send friend request';
        if (response.data is Map) {
          if (response.data['non_field_errors'] != null) {
            errorMessage = response.data['non_field_errors'][0];
          } else if (response.data['detail'] != null) {
            errorMessage = response.data['detail'];
          } else if (response.data['message'] != null) {
            errorMessage = response.data['message'];
          }
        }
        throw Exception(errorMessage);
      }
    } on FormatException {
      throw Exception('Invalid user ID format');
    } on DioException catch (e) {
      print('❌ DioException: ${e.message}'); // Debug log
      print('❌ DioException response: ${e.response?.data}'); // Debug log

      String errorMessage = 'Failed to send friend request';
      if (e.response?.data is Map) {
        if (e.response?.data['non_field_errors'] != null) {
          errorMessage = e.response?.data['non_field_errors'][0];
        } else if (e.response?.data['detail'] != null) {
          errorMessage = e.response?.data['detail'];
        } else if (e.response?.data['message'] != null) {
          errorMessage = e.response?.data['message'];
        }
      }
      throw Exception(errorMessage);
    }
  }

  // Get friend request ID for a specific user
  Future<String?> getFriendRequestIdForUser(String userId) async {
    final accessToken = await _getToken();
    if (accessToken == null) throw Exception('User is not authenticated');

    try {
      print('🔍 Finding friend request for user ID: $userId'); // Debug log
      final response =
          await _dio.get(ApiConfig.getFullUrl('friends/requests/'));

      if (response.statusCode != 200) {
        throw Exception('Failed to get friend requests');
      }

      final requests = response.data as List<dynamic>;

      // Find the pending request from or to the specified user
      final request = requests.firstWhere(
        (req) =>
            (req['from_user']['id'].toString() == userId ||
                req['to_user']['id'].toString() == userId) &&
            req['status'] == 'pending',
        orElse: () => null,
      );

      if (request == null) {
        print('❌ No pending request found for user ID: $userId'); // Debug log
        return null;
      }

      print(
        '✅ Found request ID: ${request['id']} for user ID: $userId',
      ); // Debug log
      return request['id'].toString();
    } on DioException catch (e) {
      print('❌ Error finding request: ${e.message}'); // Debug log
      throw Exception(e.response?.data?['message'] ?? 'Network error occurred');
    }
  }

  // Accept friend request by user ID or request ID
  Future<void> acceptFriendRequest(String id) async {
    final accessToken = await _getToken();
    if (accessToken == null) throw Exception('User is not authenticated');

    try {
      String requestId = id;
      print('🔍 Accepting request with ID/user ID: $id'); // Debug log

      // If this might be a user ID, try to get the request ID
      if (!id.contains('-') && int.tryParse(id) != null) {
        print(
          '🔍 ID appears to be a user ID, looking for corresponding request ID',
        ); // Debug log

        // Get all requests
        final requests = await getFriendRequests();
        print('📥 Found ${requests.length} total requests'); // Debug log

        // First, check if this is a direct request ID
        final directRequest = requests.firstWhere(
          (req) => req['id'].toString() == id,
          orElse: () => <String, dynamic>{},
        );

        if (directRequest.isNotEmpty) {
          print('✅ Found direct request with ID: $id'); // Debug log
          requestId = id;
        } else {
          // Find the pending request from the specified user
          final request = requests.firstWhere((req) {
            final fromUserId = req['from_user']['id'].toString();
            final isFromUser = fromUserId == id;
            final isPending = req['status'] == 'pending';

            print(
              '👤 Request ID: ${req['id']} - From User ID: $fromUserId - Is From User: $isFromUser - Is Pending: $isPending',
            ); // Debug log

            return isFromUser && isPending;
          }, orElse: () => <String, dynamic>{});

          if (request.isEmpty) {
            print('❌ No pending request found for user ID: $id'); // Debug log
            throw Exception('No pending friend request found for this user');
          }

          requestId = request['id'].toString();
          print('✅ Found request ID: $requestId for user ID: $id'); // Debug log
        }
      }

      print('🤝 Accepting friend request ID: $requestId'); // Debug log
      final response = await _dio.post(
        ApiConfig.getFullUrl('friends/requests/$requestId/accept/'),
      );

      print('📡 Response status: ${response.statusCode}'); // Debug log
      print('📡 Response data: ${response.data}'); // Debug log

      if (response.statusCode != 200) {
        String errorMessage = 'Failed to accept friend request';
        if (response.data is Map) {
          if (response.data['message'] != null) {
            errorMessage = response.data['message'];
          } else if (response.data['detail'] != null) {
            errorMessage = response.data['detail'];
          }
        }
        throw Exception(errorMessage);
      }
    } on DioException catch (e) {
      print('❌ Error accepting request: ${e.message}'); // Debug log
      print('❌ Error response: ${e.response?.data}'); // Debug log

      String errorMessage = 'Failed to accept friend request';
      if (e.response?.data is Map) {
        if (e.response?.data['message'] != null) {
          errorMessage = e.response?.data['message'];
        } else if (e.response?.data['detail'] != null) {
          errorMessage = e.response?.data['detail'];
        }
      }
      throw Exception(errorMessage);
    }
  }

  // Reject friend request by user ID or request ID
  Future<void> rejectFriendRequest(String id) async {
    final accessToken = await _getToken();
    if (accessToken == null) throw Exception('User is not authenticated');

    try {
      String requestId = id;
      print('🔍 Rejecting request with ID/user ID: $id'); // Debug log

      // If this might be a user ID, try to get the request ID
      if (!id.contains('-') && int.tryParse(id) != null) {
        print(
          '🔍 ID appears to be a user ID, looking for corresponding request ID',
        ); // Debug log

        // Get all requests
        final requests = await getFriendRequests();
        print('📥 Found ${requests.length} total requests'); // Debug log

        // First, check if this is a direct request ID
        final directRequest = requests.firstWhere(
          (req) => req['id'].toString() == id,
          orElse: () => <String, dynamic>{},
        );

        if (directRequest.isNotEmpty) {
          print('✅ Found direct request with ID: $id'); // Debug log
          requestId = id;
        } else {
          // Find the pending request from the specified user
          final request = requests.firstWhere((req) {
            final fromUserId = req['from_user']['id'].toString();
            final isFromUser = fromUserId == id;
            final isPending = req['status'] == 'pending';

            print(
              '👤 Request ID: ${req['id']} - From User ID: $fromUserId - Is From User: $isFromUser - Is Pending: $isPending',
            ); // Debug log

            return isFromUser && isPending;
          }, orElse: () => <String, dynamic>{});

          if (request.isEmpty) {
            print('❌ No pending request found for user ID: $id'); // Debug log
            throw Exception('No pending friend request found for this user');
          }

          requestId = request['id'].toString();
          print('✅ Found request ID: $requestId for user ID: $id'); // Debug log
        }
      }

      print('👎 Rejecting friend request ID: $requestId'); // Debug log
      final response = await _dio.post(
        ApiConfig.getFullUrl('friends/requests/$requestId/reject/'),
      );

      print('📡 Response status: ${response.statusCode}'); // Debug log
      print('📡 Response data: ${response.data}'); // Debug log

      if (response.statusCode != 200) {
        String errorMessage = 'Failed to reject friend request';
        if (response.data is Map) {
          if (response.data['message'] != null) {
            errorMessage = response.data['message'];
          } else if (response.data['detail'] != null) {
            errorMessage = response.data['detail'];
          }
        }
        throw Exception(errorMessage);
      }
    } on DioException catch (e) {
      print('❌ Error rejecting request: ${e.message}'); // Debug log
      print('❌ Error response: ${e.response?.data}'); // Debug log

      String errorMessage = 'Failed to reject friend request';
      if (e.response?.data is Map) {
        if (e.response?.data['message'] != null) {
          errorMessage = e.response?.data['message'];
        } else if (e.response?.data['detail'] != null) {
          errorMessage = e.response?.data['detail'];
        }
      }
      throw Exception(errorMessage);
    }
  }

  // Get friend requests
  Future<List<Map<String, dynamic>>> getFriendRequests() async {
    final accessToken = await _getToken();
    if (accessToken == null) throw Exception('User is not authenticated');

    try {
      print('📥 Fetching friend requests...'); // Debug log
      final response =
          await _dio.get(ApiConfig.getFullUrl('friends/requests/'));

      print('📡 Response status: ${response.statusCode}'); // Debug log
      print('📡 Response data: ${response.data}'); // Debug log

      if (response.statusCode == 200) {
        if (response.data is! List) {
          throw Exception('Invalid response format: expected a list');
        }
        final List<dynamic> requests = response.data;
        return requests.map((req) {
          if (req is! Map<String, dynamic>) {
            throw Exception('Invalid request format: expected a map');
          }
          // Ensure from_user and to_user are maps
          if (req['from_user'] is! Map<String, dynamic>) {
            throw Exception('Invalid from_user format');
          }
          if (req['to_user'] is! Map<String, dynamic>) {
            throw Exception('Invalid to_user format');
          }
          return req;
        }).toList();
      }
      throw Exception(
        response.data['message'] ?? 'Failed to fetch friend requests',
      );
    } on DioException catch (e) {
      print('❌ DioException: ${e.message}'); // Debug log
      print('❌ DioException response: ${e.response?.data}'); // Debug log
      throw Exception(e.response?.data?['message'] ?? 'Network error occurred');
    }
  }

  // Unfriend
  Future<void> unfriend(String friendshipId) async {
    final accessToken = await _getToken();
    if (accessToken == null) throw Exception('User is not authenticated');

    try {
      print('🔄 Unfriending friendship ID: $friendshipId'); // Debug log

      // The correct URL format based on the backend code
      final response = await _dio.delete(
        ApiConfig.getFullUrl('friends/friendships/$friendshipId/unfriend/'),
      );

      print('📡 Response status: ${response.statusCode}'); // Debug log
      if (response.data != null) {
        print('📡 Response data: ${response.data}'); // Debug log
      }

      if (response.statusCode != 204 && response.statusCode != 200) {
        String errorMessage = 'Failed to unfriend';
        if (response.data is Map) {
          if (response.data['message'] != null) {
            errorMessage = response.data['message'];
          } else if (response.data['detail'] != null) {
            errorMessage = response.data['detail'];
          }
        }
        throw Exception(errorMessage);
      }
    } on DioException catch (e) {
      print('❌ Error unfriending: ${e.message}'); // Debug log
      print('❌ Error response: ${e.response?.data}'); // Debug log
      print('❌ Request: ${e.requestOptions.uri}'); // Debug log
      print('❌ Headers: ${e.requestOptions.headers}'); // Debug log

      String errorMessage = 'Failed to unfriend';
      if (e.response?.data is Map) {
        if (e.response?.data['message'] != null) {
          errorMessage = e.response?.data['message'];
        } else if (e.response?.data['detail'] != null) {
          errorMessage = e.response?.data['detail'];
        }
      }
      throw Exception(errorMessage);
    }
  }

  Future<void> cancelFriendRequest(String userId) async {
    final accessToken = await _getToken();
    if (accessToken == null) throw Exception('User is not authenticated');

    try {
      // First, get all pending requests
      final response =
          await _dio.get(ApiConfig.getFullUrl('friends/requests/'));

      if (response.statusCode != 200) {
        throw Exception('Failed to get friend requests');
      }

      final requests = response.data as List<dynamic>;

      // Find the pending request where the current user is the sender and the specified user is the receiver
      final request = requests.firstWhere(
        (req) =>
            req['to_user']['id'].toString() == userId &&
            req['status'] == 'pending',
        orElse: () => null,
      );

      if (request == null) {
        throw Exception('No pending request found for this user');
      }

      // Delete the request
      final deleteResponse = await _dio.delete(
        ApiConfig.getFullUrl('friends/requests/${request['id']}/'),
      );

      if (deleteResponse.statusCode != 204 &&
          deleteResponse.statusCode != 200) {
        throw Exception('Failed to cancel friend request');
      }
    } catch (e) {
      print('Error canceling friend request: $e');
      throw Exception('Failed to cancel friend request: $e');
    }
  }

  // Remove a friend by user ID
  Future<void> removeFriend(String userId) async {
    final accessToken = await _getToken();
    if (accessToken == null) throw Exception('User is not authenticated');

    try {
      print('🔄 Removing friend with user ID: $userId'); // Debug log

      // First, get all friendships to find the one with this user
      final friendships = await getFriends();
      print('📥 Found ${friendships.length} total friendships'); // Debug log

      // Log each friendship for debugging
      for (var i = 0; i < friendships.length; i++) {
        print('👥 Friendship $i: ${friendships[i]}'); // Debug log
      }

      // Try to find the friendship that contains this user ID
      Map<String, dynamic>? matchingFriendship;

      for (var friendship in friendships) {
        // Check if this friendship contains the user we want to remove
        if (friendship['id']?.toString() == userId) {
          matchingFriendship = friendship;
          print('✅ Found friendship by id match: $matchingFriendship');
          break;
        }
      }

      // If no match found, throw an exception
      if (matchingFriendship == null) {
        print('❌ No friendship found for user ID: $userId'); // Debug log
        throw Exception('Friendship not found for this user');
      }

      // Get the friendship ID from the friendship_id field
      if (!matchingFriendship.containsKey('friendship_id')) {
        print('❌ No friendship_id found in friendship: $matchingFriendship');
        throw Exception('Friendship data is missing required ID information');
      }

      final friendshipId = matchingFriendship['friendship_id']?.toString();
      if (friendshipId == null) {
        print('❌ Friendship ID is null');
        throw Exception('Friendship ID is null');
      }

      print('✅ Using friendship_id: $friendshipId');

      // Use the unfriend method with the friendship ID
      await unfriend(friendshipId);
    } catch (e) {
      print('❌ Error removing friend: $e'); // Debug log
      throw Exception('Failed to remove friend: $e');
    }
  }

  // Direct unfriend by friendship ID (for the Unknown user case)
  Future<void> removeUnknownFriend() async {
    final accessToken = await _getToken();
    if (accessToken == null) throw Exception('User is not authenticated');

    try {
      print('🔄 Getting friendships to find Unknown user'); // Debug log

      // Get all friendships
      final friendships = await getFriends();
      print('📥 Found ${friendships.length} total friendships'); // Debug log

      // Log each friendship for debugging
      for (var i = 0; i < friendships.length; i++) {
        print('👥 Friendship $i: ${friendships[i]}'); // Debug log
      }

      // Find the Unknown user friendship
      Map<String, dynamic>? unknownFriendship;
      for (var friendship in friendships) {
        if (friendship['name'] == 'Unknown' ||
            friendship['username'] == 'Unknown') {
          unknownFriendship = friendship;
          break;
        }
      }

      if (unknownFriendship == null) {
        throw Exception('Unknown friend not found');
      }

      print('✅ Found Unknown friendship: $unknownFriendship'); // Debug log

      // Get the friendship ID
      String? friendshipId;
      if (unknownFriendship.containsKey('friendship_id')) {
        friendshipId = unknownFriendship['friendship_id'].toString();
      } else if (unknownFriendship.containsKey('id')) {
        friendshipId = unknownFriendship['id'].toString();
      }

      if (friendshipId == null) {
        throw Exception('Could not find friendship ID for Unknown user');
      }

      print(
        '🔄 Unfriending Unknown user with friendship ID: $friendshipId',
      ); // Debug log

      // Use the unfriend method with the friendship ID
      await unfriend(friendshipId);
    } catch (e) {
      print('❌ Error removing Unknown friend: $e'); // Debug log
      throw Exception('Failed to remove Unknown friend: $e');
    }
  }
}
