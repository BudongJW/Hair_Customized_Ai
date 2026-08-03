import React from "react";
import { StyleSheet, View } from "react-native";
import { colors, radius, shadow, spacing } from "../theme/tokens";

export function Panel({ children, style }) {
  return <View style={[styles.panel, style]}>{children}</View>;
}

const styles = StyleSheet.create({
  panel: {
    backgroundColor: colors.panel,
    borderColor: colors.line,
    borderRadius: radius.md,
    borderWidth: 1,
    padding: spacing.md,
    gap: spacing.md,
    ...shadow
  }
});
