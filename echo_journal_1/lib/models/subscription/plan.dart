class Plan {
  final int id;
  final String name;
  final String planType;
  final double price;
  final int durationDays;
  final String description;
  final Map<String, dynamic> features;
  final bool isActive;
  final DateTime createdAt;
  final DateTime updatedAt;

  Plan({
    required this.id,
    required this.name,
    required this.planType,
    required this.price,
    required this.durationDays,
    required this.description,
    required this.features,
    required this.isActive,
    required this.createdAt,
    required this.updatedAt,
  });

  factory Plan.fromJson(Map<String, dynamic> json) {
    return Plan(
      id: json['id'],
      name: json['name'],
      planType: json['plan_type'],
      price: json['price'].toDouble(),
      durationDays: json['duration_days'],
      description: json['description'],
      features: json['features'],
      isActive: json['is_active'],
      createdAt: DateTime.parse(json['created_at']),
      updatedAt: DateTime.parse(json['updated_at']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'plan_type': planType,
      'price': price,
      'duration_days': durationDays,
      'description': description,
      'features': features,
      'is_active': isActive,
      'created_at': createdAt.toIso8601String(),
      'updated_at': updatedAt.toIso8601String(),
    };
  }
}
