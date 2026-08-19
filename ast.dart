import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:analyzer/dart/analysis/utilities.dart';
import 'package:analyzer/dart/analysis/features.dart';

void main(List<String> args) async {
  final time = DateTime.now().millisecondsSinceEpoch;
  // args = ['lib/main.dart'];
  if (args.isEmpty) {
    // Default test offsets for Column in lib/main.dart
    await astProvider(["lib/main.dart", '1574', '1595']);
  } else {
    await astProvider(args);
  }
  print("Time taken: ${DateTime.now().millisecondsSinceEpoch - time}");
}

Future<void> astProvider(List<String> args, {String? sdkPath}) async {
  String rawPath = 'lib/main.dart';
  int? targetOffset;
  int? targetEnd;

  if (args.isNotEmpty) {
    rawPath = args[0];
    if (args.length >= 3) {
      targetOffset = int.tryParse(args[1]);
      targetEnd = int.tryParse(args[2]);
    }
  }

  final String absolutePath = p.canonicalize(p.absolute(rawPath));
  final file = File(absolutePath);

  if (!file.existsSync()) {
    print('❌ Error: File not found at $absolutePath');
    return;
  }

  if (targetOffset == null || targetEnd == null) {
    await getWidgetHierarchy(absolutePath);
  } else {
    await getWidgetDetails(
      absolutePath,
      targetOffset,
      targetEnd,
      sdkPath: sdkPath,
    );
  }
}

Future<void> getWidgetHierarchy(String absolutePath) async {
  try {
    final result = parseFile(
      path: absolutePath,
      featureSet: FeatureSet.latestLanguageVersion(),
    );

    final visitor = WidgetHierarchyVisitor(includeDetails: false);
    result.unit.visitChildren(visitor);
    print(JsonEncoder.withIndent('  ').convert(visitor.widgets));
  } catch (e) {
    print('❌ Error in Hierarchy Mode: $e');
  }
}

Future<void> getWidgetDetails(
  String absolutePath,
  int targetOffset,
  int targetEnd, {
  String? sdkPath,
}) async {
  try {
    // MODE 2: Ultra-Fast Detail (Unresolved)
    final result = parseFile(
      path: absolutePath,
      featureSet: FeatureSet.latestLanguageVersion(),
    );

    final rangeVisitor = _RangeSearchVisitor(targetOffset, targetEnd);
    result.unit.visitChildren(rangeVisitor);

    final targetNode = rangeVisitor.bestNode;
    if (targetNode == null) {
      print('[]');
      return;
    }

    final visitor = WidgetHierarchyVisitor(
      targetOffset: targetNode.offset,
      targetEnd: targetNode.end,
      includeDetails: true,
    );

    // In unresolved mode, we extract what we can from the AST node itself.
    final data = visitor.manualParse(targetNode);

    // Fast Definition Look-up for available_arguments
    final crawler = FastDefinitionCrawler();
    data['available_arguments'] = await crawler.getAvailableArguments(
      data['widget'],
      absolutePath,
    );

    print(JsonEncoder.withIndent('  ').convert([data]));
  } catch (e) {
    print('❌ Error in Detail Mode: $e');
  }
}

class PackageResolver {
  static Map<String, String>? _packageMap;

  static Future<void> _init(String currentPath) async {
    if (_packageMap != null) return;
    try {
      final configPath = p.join(
        p.dirname(p.dirname(currentPath)),
        '.dart_tool',
        'package_config.json',
      );
      final file = File(configPath);
      if (file.existsSync()) {
        final data = jsonDecode(file.readAsStringSync());
        _packageMap = {};
        for (var pkg in data['packages']) {
          String root = pkg['rootUri'];
          if (root.startsWith('file:///')) {
            root = Uri.parse(root).toFilePath();
          } else if (root.startsWith('../')) {
            root = p.canonicalize(p.join(p.dirname(configPath), root));
          }
          _packageMap![pkg['name']] = root;
        }
      }
    } catch (_) {}
  }

  static String? resolve(String uri, String currentPath) {
    if (uri.startsWith('package:')) {
      final parts = uri.substring(8).split('/');
      final pkgName = parts.first;
      final relativePath = parts.skip(1).join('/');
      final pkgRoot = _packageMap?[pkgName];
      if (pkgRoot != null) {
        return p.join(pkgRoot, 'lib', relativePath);
      }
    } else if (!uri.startsWith('dart:')) {
      return p.canonicalize(p.join(p.dirname(currentPath), uri));
    }
    return null;
  }
}

class FastDefinitionCrawler {
  static final Map<String, List<Map<String, dynamic>>> _cache = {};

  Future<List<Map<String, dynamic>>> getAvailableArguments(
    String widgetName,
    String currentFilePath,
  ) async {
    // 0. Fallback for common Flutter widgets (Immediate)
    if (widgetName == 'Column' || widgetName == 'Row') {
      return [
        {
          'name': 'mainAxisAlignment',
          'type': 'MainAxisAlignment',
          'isRequired': false,
          'isNamed': true,
          'defaultValue': 'MainAxisAlignment.start',
        },
        {
          'name': 'mainAxisSize',
          'type': 'MainAxisSize',
          'isRequired': false,
          'isNamed': true,
          'defaultValue': 'MainAxisSize.max',
        },
        {
          'name': 'crossAxisAlignment',
          'type': 'CrossAxisAlignment',
          'isRequired': false,
          'isNamed': true,
          'defaultValue': 'CrossAxisAlignment.center',
        },
        {
          'name': 'verticalDirection',
          'type': 'VerticalDirection',
          'isRequired': false,
          'isNamed': true,
          'defaultValue': 'VerticalDirection.down',
        },
        {
          'name': 'textDirection',
          'type': 'TextDirection?',
          'isRequired': false,
          'isNamed': true,
          'defaultValue': 'null',
        },
        {
          'name': 'textBaseline',
          'type': 'TextBaseline?',
          'isRequired': false,
          'isNamed': true,
          'defaultValue': 'null',
        },
        {
          'name': 'children',
          'type': 'List<Widget>',
          'isRequired': false,
          'isNamed': true,
          'defaultValue': 'const <Widget>[]',
        },
      ];
    }

    if (_cache.containsKey(widgetName)) return _cache[widgetName]!;

    await PackageResolver._init(currentFilePath);

    // 1. Search in current file
    final file = File(currentFilePath);
    if (file.existsSync()) {
      final content = file.readAsStringSync();
      final localArgs = _searchInFile(content, widgetName);
      if (localArgs.isNotEmpty) {
        _cache[widgetName] = localArgs;
        return localArgs;
      }

      // 2. Identify imports and search them recursively
      final imports = _extractImports(content);
      for (var imp in imports) {
        final path = PackageResolver.resolve(imp, currentFilePath);
        if (path != null && File(path).existsSync()) {
          final found = await _findInPackage(path, widgetName, depth: 3);
          if (found != null) {
            _cache[widgetName] = found;
            return found;
          }
        }
      }
    }

    return [];
  }

  Future<List<Map<String, dynamic>>?> _findInPackage(
    String startPath,
    String widgetName, {
    int depth = 5,
  }) async {
    if (depth <= 0) return null;

    final file = File(startPath);
    if (!file.existsSync()) return null;

    final content = file.readAsStringSync();
    if (content.contains('class $widgetName')) {
      final res = _searchInFile(content, widgetName);
      if (res.isNotEmpty) return res;
    }

    // Check exports
    final exports = _extractExports(content);

    // Heuristic: Prioritize exports that likely contain the widget
    final prioritized = exports.where((e) {
      final lowerName = widgetName.toLowerCase();
      final lowerExp = e.toLowerCase();
      return lowerExp.contains(lowerName) ||
          lowerName.contains(
            lowerExp
                .split('/')
                .last
                .replaceAll('.dart', '')
                .replaceAll('_', ''),
          );
    }).toList();

    final otherExports = exports
        .where((e) => !prioritized.contains(e))
        .toList();

    for (var exp in [...prioritized, ...otherExports]) {
      final path = PackageResolver.resolve(exp, startPath);
      if (path != null && File(path).existsSync()) {
        final res = await _findInPackage(path, widgetName, depth: depth - 1);
        if (res != null) return res;
      }
    }

    return null;
  }

  List<String> _extractExports(String content) {
    final List<String> exports = [];
    final lines = content.split('\n');
    for (var line in lines) {
      final trimmed = line.trim();
      if (trimmed.startsWith('export ')) {
        final match = RegExp(
          "export\\s+['\"]([^'\"]+)['\"]",
        ).firstMatch(trimmed);
        if (match != null) exports.add(match.group(1)!);
      }
    }
    return exports;
  }

  List<String> _extractImports(String content) {
    final List<String> imports = [];
    final lines = content.split('\n');
    for (var line in lines) {
      final trimmed = line.trim();
      if (trimmed.startsWith('import ')) {
        final match = RegExp(
          "import\\s+['\"]([^'\"]+)['\"]",
        ).firstMatch(trimmed);
        if (match != null) imports.add(match.group(1)!);
      }
    }
    return imports;
  }

  List<Map<String, dynamic>> _searchInFile(String content, String widgetName) {
    try {
      final res = parseString(
        content: content,
        featureSet: FeatureSet.latestLanguageVersion(),
        throwIfDiagnostics: false,
      );

      for (var declaration in res.unit.declarations) {
        if (declaration is ClassDeclaration &&
            declaration.name.lexeme == widgetName) {
          for (var member in declaration.members) {
            if (member is ConstructorDeclaration) {
              if (member.name == null || member.name!.lexeme == widgetName) {
                return member.parameters.parameters.map((p) {
                  return {
                    'name': p.name?.lexeme ?? '',
                    'type': p.toSource().split(' ').first,
                    'isRequired': p.isRequired,
                    'isNamed': p.isNamed,
                    'defaultValue': p is DefaultFormalParameter
                        ? p.defaultValue?.toSource()
                        : null,
                  };
                }).toList();
              }
            }
          }
        }
      }
    } catch (_) {}
    return [];
  }
}

class WidgetHierarchyVisitor extends RecursiveAstVisitor<void> {
  final int? targetOffset;
  final int? targetEnd;
  final bool includeDetails;
  final List<Map<String, dynamic>> widgets = [];

  // Keep track of nodes that are already processed as children to avoid duplication at top level
  final Set<Expression> _processedWidgets = {};

  WidgetHierarchyVisitor({
    this.targetOffset,
    this.targetEnd,
    required this.includeDetails,
  });

  @override
  void visitInstanceCreationExpression(InstanceCreationExpression node) {
    _handlePotentialWidget(node);
  }

  @override
  void visitMethodInvocation(MethodInvocation node) {
    final String name = node.methodName.name;
    // Heuristic: If it starts with Uppercase, it's likely a widget in unresolved mode
    if (name.isNotEmpty &&
        name[0].toUpperCase() == name[0] &&
        name != 'Theme' &&
        name != 'MediaQuery') {
      _handlePotentialWidget(node);
    } else {
      node.visitChildren(this);
    }
  }

  void _handlePotentialWidget(Expression node) {
    if (includeDetails) {
      if (node.offset == targetOffset && node.end == targetEnd) {
        widgets.add(_parseWidget(node, includeDetails: true));
        return;
      }
      node.visitChildren(this);
    } else {
      if (!_processedWidgets.contains(node)) {
        widgets.add(_parseWidget(node, includeDetails: false));
      } else {
        node.visitChildren(this);
      }
    }
  }

  Map<String, dynamic> _parseWidget(
    Expression node, {
    required bool includeDetails,
  }) {
    // Mark as processed
    _processedWidgets.add(node);

    String widgetName = '';
    ArgumentList? argList;

    if (node is InstanceCreationExpression) {
      widgetName = node.constructorName.type.toSource();
      argList = node.argumentList;
    } else if (node is MethodInvocation) {
      widgetName = node.methodName.name;
      argList = node.argumentList;
    }

    final Map<String, dynamic> data = {
      'widget': widgetName,
      'offset': node.offset,
      'end': node.end,
    };

    if (includeDetails) {
      if (node is InstanceCreationExpression) {
        final element = node.constructorName.element;
        data['type'] = node.staticType?.getDisplayString() ?? widgetName;
        data['available_arguments'] =
            element?.formalParameters
                .map(
                  (p) => {
                    'name': p.name,
                    'type': p.type.getDisplayString(),
                    'isRequired': p.isRequired,
                    'isNamed': p.isNamed,
                    'defaultValue': p.defaultValueCode,
                  },
                )
                .toList() ??
            [];
      } else {
        data['type'] = widgetName;
        data['available_arguments'] = [];
      }

      final providedArgs = <String, dynamic>{};
      if (argList != null) {
        for (var arg in argList.arguments) {
          if (arg is NamedExpression) {
            providedArgs[arg.name.label.name] = arg.expression.toSource();
          } else {
            providedArgs['positional_${argList.arguments.indexOf(arg)}'] = arg
                .toSource();
          }
        }
      }
      data['provided_arguments'] = providedArgs;

      // In Detail Mode, we DO NOT include children
      data['children'] = [];
    } else {
      // Recursively find children only in Hierarchy Mode
      final children = <Map<String, dynamic>>[];
      if (argList != null) {
        for (var arg in argList.arguments) {
          final Expression expr = arg is NamedExpression ? arg.expression : arg;
          _collectWidgetsFromExpression(expr, children, includeDetails);
        }
      }
      data['children'] = children;
    }

    return data;
  }

  void _collectWidgetsFromExpression(
    Expression expr,
    List<Map<String, dynamic>> children,
    bool details,
  ) {
    if (expr is InstanceCreationExpression) {
      children.add(_parseWidget(expr, includeDetails: details));
    } else if (expr is MethodInvocation) {
      final String name = expr.methodName.name;
      if (name.isNotEmpty &&
          name[0].toUpperCase() == name[0] &&
          name != 'Theme' &&
          name != 'MediaQuery') {
        children.add(_parseWidget(expr, includeDetails: details));
      } else {
        expr.accept(_NestedWidgetFinder(this, children, details));
      }
    } else if (expr is ListLiteral) {
      for (var element in expr.elements) {
        if (element is Expression) {
          _collectWidgetsFromExpression(element, children, details);
        }
      }
    } else {
      // Use a localized visitor to find any InstanceCreationExpressions buried in the expression
      expr.accept(_NestedWidgetFinder(this, children, details));
    }
  }

  Map<String, dynamic> manualParse(Expression node) =>
      _parseWidget(node, includeDetails: true);
}

class _NestedWidgetFinder extends RecursiveAstVisitor<void> {
  final WidgetHierarchyVisitor parent;
  final List<Map<String, dynamic>> children;
  final bool includeDetails;

  _NestedWidgetFinder(this.parent, this.children, this.includeDetails);

  @override
  void visitInstanceCreationExpression(InstanceCreationExpression node) {
    children.add(parent._parseWidget(node, includeDetails: includeDetails));
  }

  @override
  void visitMethodInvocation(MethodInvocation node) {
    final String name = node.methodName.name;
    if (name.isNotEmpty &&
        name[0].toUpperCase() == name[0] &&
        name != 'Theme' &&
        name != 'MediaQuery') {
      children.add(parent._parseWidget(node, includeDetails: includeDetails));
    } else {
      super.visitMethodInvocation(node);
    }
  }
}

/// Visitor that finds the MOST specific (innermost) widget that overlaps with the given range.
class _RangeSearchVisitor extends GeneralizingAstVisitor<void> {
  final int offset;
  final int end;
  Expression? bestNode;
  int minDistance = -1;

  _RangeSearchVisitor(this.offset, this.end);

  @override
  void visitNode(AstNode node) {
    // If this node contains the range, it's a candidate.
    bool isWidget = false;
    if (node is InstanceCreationExpression) {
      isWidget = true;
    } else if (node is MethodInvocation) {
      final name = node.methodName.name;
      if (name.isNotEmpty &&
          name[0].toUpperCase() == name[0] &&
          name != 'Theme' &&
          name != 'MediaQuery') {
        isWidget = true;
      }
    }

    if (isWidget) {
      final nodeOffset = node.offset;
      final nodeEnd = node.end;

      // Calculate distance heuristic. Lower is better.
      final distance = (nodeOffset - offset).abs() + (nodeEnd - end).abs();

      // We focus on nodes that have some relationship with the range
      // 1. Node contains selection
      // 2. Selection contains node
      // 3. Significant overlap (midpoint is inside)
      bool relevant = false;
      if (nodeOffset <= offset && nodeEnd >= end)
        relevant = true;
      else if (nodeOffset >= offset && nodeEnd <= end)
        relevant = true;
      else {
        final mid = (offset + end) ~/ 2;
        if (nodeOffset <= mid && nodeEnd >= mid) relevant = true;
      }

      if (relevant) {
        if (minDistance == -1 || distance < minDistance) {
          minDistance = distance;
          bestNode = node as Expression;
        } else if (distance == minDistance) {
          // If distances are equal, pick the smaller (deeper) node
          if (bestNode != null &&
              (nodeEnd - nodeOffset) < (bestNode!.end - bestNode!.offset)) {
            bestNode = node as Expression;
          }
        }
      }
    }

    super.visitNode(node);
  }
}
