import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { colors, radius, spacing } from "../theme/tokens";

const palette = {
  PENDING: [colors.softGold, colors.danger],
  PROCESSING: [colors.softTeal, colors.teal],
  COMPLETED: [colors.softTeal, colors.green],
  FAILED: [colors.softCoral, colors.coral],
  REJECTED: [colors.softCoral, colors.coral]
};

export function StatusPill({ status = "READY" }) {
  const [backgroundColor, color] = palette[status] || ["#EEF2F6", colors.muted];
  return (
    <View style={[styles.pill, { backgroundColor }]}>
      <Text style={[styles.text, { color }]}>{status}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  pill: {
    alignSelf: "flex-start",
    borderRadius: radius.sm,
    paddingHorizontal: spacing.sm,
    paddingVertical: 5
  },
  text: {
    fontSize: 12,
    fontWeight: "800"
  }
});
