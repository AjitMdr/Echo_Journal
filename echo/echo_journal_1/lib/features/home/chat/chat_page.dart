import 'package:echo_journal1/core/configs/theme/curved-pattern.dart';
import 'package:echo_journal1/features/home/chat/conversations_page.dart';
import 'package:echo_journal1/services/chat/chat_service.dart';
import 'package:echo_journal1/utils/toast_helper.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:echo_journal1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal1/features/widgets/navbar.dart';

class ChatPage extends StatefulWidget {
  final bool isDarkMode;

  const ChatPage({super.key, this.isDarkMode = false});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage>
    with AutomaticKeepAliveClientMixin {
  final ChatService _chatService = ChatService();
  bool _isLoading = false;
  bool _hasError = false;
  String? _errorMessage;

  @override
  bool get wantKeepAlive => true;

  @override
  void initState() {
    super.initState();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Refresh data when the page becomes visible
    _refreshData();
  }

  Future<void> _refreshData() async {
    // Refresh unread message count in the parent NavBar
    final navBarState = context.findAncestorStateOfType<NavBarState>();
    if (navBarState != null) {
      _loadUnreadMessageCount();
    }
  }

  Future<void> _loadUnreadMessageCount() async {
    try {
      await _chatService.getUnreadMessageCount();
      // The count will be updated in the NavBar through its own timer
    } catch (e) {
      debugPrint('Error loading unread message count: $e');
    }
  }

  @override
  void dispose() {
    _chatService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);

    final backgroundColor = widget.isDarkMode ? Colors.black : Colors.white;
    final textColor = widget.isDarkMode ? Colors.white : Colors.black;

    return Scaffold(
      backgroundColor: backgroundColor,
      body: Stack(
        children: [
          CustomPaint(
            painter: CurvedPatternPainter(),
            size: MediaQuery.of(context).size,
          ),
          Column(
            children: [
              // Header
              Container(
                color: Theme.of(context).scaffoldBackgroundColor,
                padding: EdgeInsets.only(
                  left: 12.0,
                  right: 12.0,
                  top: 0,
                  bottom: 0,
                ),
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          'Chat',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: textColor,
                          ),
                        ),
                        IconButton(
                          icon: Icon(Icons.refresh, color: textColor),
                          onPressed: () {
                            _refreshData();
                          },
                          tooltip: 'Refresh',
                          padding: EdgeInsets.all(8),
                          constraints: BoxConstraints(),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              // Direct Messages Content
              Expanded(child: ConversationsPage(isDarkMode: widget.isDarkMode)),
            ],
          ),
        ],
      ),
    );
  }
}
