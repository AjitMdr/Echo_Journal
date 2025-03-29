import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:echo_journal_1/features/widgets/streak_badge_widget.dart';
import 'package:provider/provider.dart';
import 'package:echo_journal_1/core/configs/theme/theme-provider.dart';

class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  _ProfilePageState createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  String? username = "John Doe"; // Static username for now
  String? email = "john@example.com"; // Static email for now
  String? profilePictureUrl; // Will use default image
  bool isLoading = false;

  // To pick an image for profile picture
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    // Removed _loadProfileData() call since we're using static data
  }

  // Load profile data from cache or API
  Future<void> _loadProfileData() async {
    // Commented out for now - will implement service calls later
    /*setState(() {
      isLoading = true;
    });

    try {
      final profileData = await ProfileService.getCachedProfile();
      if (profileData != null) {
        setState(() {
          username = profileData['username'];
          email = profileData['email'];
          profilePictureUrl = profileData['profile_picture'];
        });
      } else {
        final profile = await ProfileService.fetchProfile();
        setState(() {
          username = profile['username'];
          email = profile['email'];
          profilePictureUrl = profile['profile_picture'];
        });
      }
    } catch (e) {
      print('Error loading profile: $e');
    } finally {
      setState(() {
        isLoading = false;
      });
    }*/
  }

  // Method to pick and update profile picture
  Future<void> _updateProfilePicture() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      // Commented out service call for now
      /*setState(() {
        isLoading = true;
      });
      try {
        File imageFile = File(pickedFile.path);
        await ProfileService.updateProfilePicture(imageFile);
        await _loadProfileData();
      } catch (e) {
        print('Error updating profile picture: $e');
      } finally {
        setState(() {
          isLoading = false;
        });
      }*/
    }
  }

  // Method to update user profile
  Future<void> _updateProfile() async {
    if (username == null || email == null) {
      print('Please fill in all fields');
      return;
    }

    // Commented out service call for now
    /*setState(() {
      isLoading = true;
    });

    try {
      final updatedProfileData = {'username': username, 'email': email};

      await ProfileService.updateProfile(updatedProfileData);
      await _loadProfileData();
    } catch (e) {
      print('Error updating profile: $e');
    } finally {
      setState(() {
        isLoading = false;
      });
    }*/
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDarkMode = themeProvider.isDarkMode;

    return Scaffold(
      appBar: AppBar(title: Text('Profile')),
      body: isLoading
          ? Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              child: Padding(
                padding: EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Profile Picture
                    Center(
                      child: GestureDetector(
                        onTap: _updateProfilePicture,
                        child: CircleAvatar(
                          radius: 60,
                          backgroundImage: profilePictureUrl != null
                              ? NetworkImage(profilePictureUrl!)
                              : AssetImage('assets/default_profile.png')
                                  as ImageProvider,
                        ),
                      ),
                    ),
                    SizedBox(height: 20),
                    // Streak and Badges
                    StreakBadgeWidget(isDarkMode: isDarkMode),
                    SizedBox(height: 20),
                    // Username Field
                    TextField(
                      decoration: InputDecoration(
                        labelText: 'Username',
                        border: OutlineInputBorder(),
                      ),
                      onChanged: (value) {
                        setState(() {
                          username = value;
                        });
                      },
                      controller: TextEditingController(text: username),
                    ),
                    SizedBox(height: 10),
                    // Email Field
                    TextField(
                      decoration: InputDecoration(
                        labelText: 'Email',
                        border: OutlineInputBorder(),
                      ),
                      onChanged: (value) {
                        setState(() {
                          email = value;
                        });
                      },
                      controller: TextEditingController(text: email),
                    ),
                    SizedBox(height: 20),
                    // Update Profile Button
                    Center(
                      child: ElevatedButton(
                        onPressed: _updateProfile,
                        child: Text('Update Profile'),
                      ),
                    ),
                  ],
                ),
              ),
            ),
    );
  }
}
