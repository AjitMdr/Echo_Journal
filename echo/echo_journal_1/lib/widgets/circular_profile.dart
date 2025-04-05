import 'package:flutter/material.dart';
import 'package:cached_network_image/cached_network_image.dart';
import '../models/user.dart';

class CircularProfile extends StatelessWidget {
  final User user;
  final double size;
  final bool showName;
  final bool showStatus;
  final VoidCallback? onTap;

  const CircularProfile({
    super.key,
    required this.user,
    this.size = 60,
    this.showName = true,
    this.showStatus = true,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: size,
            height: size,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(
                color: Theme.of(context).primaryColor,
                width: 2,
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 8,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: ClipOval(
              child: user.profileImage != null
                  ? CachedNetworkImage(
                      imageUrl: user.profileImage!,
                      fit: BoxFit.cover,
                      width: size,
                      height: size,
                      memCacheWidth: size.toInt(),
                      memCacheHeight: size.toInt(),
                      maxWidthDiskCache: size.toInt(),
                      maxHeightDiskCache: size.toInt(),
                      placeholder: (context, url) => Container(
                        color: Colors.grey[200],
                        width: size,
                        height: size,
                        child: const Center(
                          child: CircularProgressIndicator(),
                        ),
                      ),
                      errorWidget: (context, url, error) => Container(
                        color: Colors.grey[200],
                        width: size,
                        height: size,
                        child: Icon(
                          Icons.person,
                          size: size * 0.5,
                          color: Colors.grey[400],
                        ),
                      ),
                    )
                  : Container(
                      color: Colors.grey[200],
                      width: size,
                      height: size,
                      child: Icon(
                        Icons.person,
                        size: size * 0.5,
                        color: Colors.grey[400],
                      ),
                    ),
            ),
          ),
          if (showName) ...[
            const SizedBox(height: 8),
            Text(
              user.username,
              style: Theme.of(context).textTheme.titleMedium,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ],
          if (showStatus && user.isOnline != null) ...[
            const SizedBox(height: 4),
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: user.isOnline! ? Colors.green : Colors.grey,
                border: Border.all(color: Colors.white, width: 2),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
